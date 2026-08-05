#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shlex
import signal
import sys
from collections import defaultdict
from importlib import import_module
from inspect import isfunction
from pathlib import Path
from typing import Any

import click

from revng.internal.support import cache_directory
from revng.internal.support.collect import collect_files_recursive
from revng.pypeline.cli.project import project
from revng.pypeline.cli.wrappers import WRAPPER_REGISTRY, WrapperOption
from revng.pypeline.main import main as pype_main
from revng.pypeline.main import pype
from revng.support import collect_files, get_root

from .common import ClickContext, CommandRegistry, ContextObject, WrappableCommand, cli_logger
from .common import pass_context
from .support import executable_name, is_file_executable, search_prefixes


class ValgrindWrapperOption(WrapperOption):
    def generate_prefix(self, value: Any) -> list[str]:
        suppressions = collect_files([get_root()], ["share", "revng"], "*.supp")
        return ["valgrind", *(f"--suppressions={s}" for s in suppressions)]


class WrapperWrapperOption(WrapperOption):
    def __init__(self, name: str, help: str):  # noqa: A002
        super().__init__(name, help, type_=str)

    def generate_prefix(self, value: Any) -> list[str]:
        return shlex.split(value)


WRAPPER_REGISTRY.register_wrappers(
    WrapperOption(
        name="perf",
        help="Run program(s) under perf (for use with hotspot).",
        prefix=["perf", "record", "--call-graph", "dwarf", "--output=perf.data"],
    ),
    WrapperOption("heaptrack", help="Run program(s) under heaptrack.", prefix=["heaptrack"]),
    WrapperOption("gdb", help="Run program(s) under gdb.", prefix=["gdb", "-q", "--args"]),
    WrapperOption("lldb", help="Run program(s) under lldb.", prefix=["lldb", "--"]),
    ValgrindWrapperOption("valgrind", help="Run program(s) under valgrind."),
    WrapperOption(
        "callgrind",
        help="Run program(s) under callgrind.",
        prefix=["valgrind", "--tool=callgrind"],
    ),
    WrapperOption("rr", help="Run program(s) under rr.", prefix=["rr"]),
    WrapperWrapperOption("wrapper", help="Run program(s) with the specified wrapper."),
)


class GroupRegistry(CommandRegistry):
    """
    Registry of the click groups making up the revng command-line, addressed by
    their path, e.g. `("model", "import")` for `revng model import`; the root
    command is addressed by the empty tuple.
    """

    def __init__(self, root: click.Group):
        self.groups: dict[tuple[str, ...], tuple[click.Group, str]] = {}
        # Commands whose group has not been registered yet
        self.pending: dict[tuple[str, ...], list[click.Command]] = defaultdict(list)
        self._add_group((), root)

    def _add_group(self, path: tuple[str, ...], group: click.Group):
        assert path not in self.groups
        self.groups[path] = (group, "-".join(path))

        # Recursively check for sub-groups
        for name, command in group.commands.items():
            if isinstance(command, click.Group):
                self._add_group((*path, name), command)

        for command in self.pending.pop(path, []):
            self.register(path, command)

    def register(self, group: tuple[str, ...], command: click.Command):
        if group not in self.groups:
            self.pending[group].append(command)
            return

        self.groups[group][0].add_command(command)
        if isinstance(command, click.Group):
            assert command.name is not None
            self._add_group((*group, command.name), command)

    def check(self):
        if len(self.pending) > 0:
            cli_logger.log("Commands registered in non-existing group(s):")
            for group, commands in self.pending.items():
                for command in commands:
                    cli_logger.log(f"* {command} in {group}")
            raise ValueError

    @staticmethod
    def _build_external_command(group: tuple[str, ...], name: str, path: str) -> click.Command:
        @click.command(
            cls=WrappableCommand,
            name=name,
            help=f"see {" ".join((executable_name(), *group, name))} --help",
            add_help_option=False,
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        )
        @click.argument("arguments", metavar="[ARGS]...", nargs=-1, type=click.UNPROCESSED)
        @pass_context
        def external_command(ctx: ClickContext, arguments: tuple[str, ...]) -> int:
            return ctx.obj.try_run([path, *arguments])

        return external_command

    def _resolve_command_path(self, executable: str) -> tuple[tuple[str, ...], str]:
        """
        Turn the path of an executable, relative to `libexec/revng`, into the group
        it belongs to and its command name. Each directory is a group, then the
        longest prefix of `-`-separated words matching a group is consumed.
        """
        name = Path(executable).parts[-1]
        group = Path(executable).parts[:-1]

        def is_group_prefix(group_tuple: tuple[str, ...]):
            return len(group_tuple) > len(group) and group_tuple[: len(group)] == group

        candidates = [
            (k, v[1])
            for k, v in self.groups.items()
            if v[1] != "" and name.startswith(v[1]) and name != v[1] and is_group_prefix(k)
        ]
        candidates.sort(key=lambda x: len(x[1]), reverse=True)

        if len(candidates) > 0:
            group = candidates[0][0]
            name = name.removeprefix(f"{candidates[0][1]}-")

        return group, name

    def add_external_executable(self, path: str, executable: str):
        group, name = self._resolve_command_path(executable)
        if name not in self.groups[group][0].commands:
            self.register(group, self._build_external_command(group, name, path))


def patch_pype():
    """
    revng is based on `pype`, but we want to change some defaults to be revng specific,
    and we want to add some commands.
    """

    # Replace the name (needed for autocompletion and usage)
    pype.name = "revng"

    # Patch the callback function so that the cli_logger is enabled with
    # `--verbose`
    pype_original_callback = pype.callback

    def pype_callback(*args, **kwargs):
        if kwargs["verbose"]:
            cli_logger.debug = True
        return pype_original_callback(*args, **kwargs)

    pype.callback = pype_callback

    # Replace the default for pipebox
    for param in pype.params:
        if param.name == "pipebox":
            param.default = Path(__file__).parent.parent / "pipebox.py"

    # Change the default for pipeline
    for param in project.params:
        if param.name == "pipeline":
            param.default = Path(__file__).parent.parent / "pipeline.yml"
        elif param.name == "cache_dir":
            param.default = str(cache_directory())
        elif param.name == "storage_provider":
            param.envvar = ["REVNG_STORAGE_PROVIDER", param.envvar]


def register_external_commands(registry: GroupRegistry):
    """
    Register the executables in `libexec/revng` as commands. Their name is
    split on `-` to find the innermost group they belong to, e.g. `model-opt`
    becomes `model opt`.
    """
    for executable, path in collect_files_recursive(search_prefixes(), ["libexec", "revng"], "*"):
        if is_file_executable(path) and os.path.splitext(path)[1] == "":
            registry.add_external_executable(path, executable)


def load_commands(registry: CommandRegistry):
    """Let each module in `_commands` register the commands it implements."""
    modules = []
    with os.scandir(Path(__file__).parent / "_commands") as scan:
        for entry in scan:
            entry_path = Path(entry.path)
            if entry_path.name.startswith("__") or entry_path.name.startswith("."):
                continue
            if entry.is_file():
                modules.append(import_module(f"._commands.{entry_path.stem}", __package__))
            elif entry.is_dir():
                modules.append(import_module(f"._commands.{entry_path.name}", __package__))

    for module in modules:
        setup = getattr(module, "setup", None)
        if setup is not None and isfunction(setup):
            setup(registry)


def main():
    """Entry point for revng"""
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(1))
    patch_pype()

    # Create and populate the registry
    registry = GroupRegistry(pype)
    load_commands(registry)
    register_external_commands(registry)
    registry.check()

    pype_main(sys.argv[1:], ContextObject)


if __name__ == "__main__":
    main()
