#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
This is just a wrapper over `pype` that sets pipebox to the revng pipebox path.
The path is computed relatively to this file, so this should work regardless of
where revng is installed.
"""

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
from revng.pypeline.cli.project import project
from revng.pypeline.cli.wrappers import WRAPPER_REGISTRY, WrapperOption
from revng.pypeline.main import main as pype_main
from revng.pypeline.main import pype
from revng.support import collect_files, get_root

from .common import CommandRegistry, ContextObject, cli_logger


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
    their path, e.g. `("model", "import")` for `revng2 model import`; the root
    command is addressed by the empty tuple.
    """

    def __init__(self, root: click.Group):
        self.groups: dict[tuple[str, ...], click.Group] = {}
        # Commands whose group has not been registered yet
        self.pending: dict[tuple[str, ...], list[click.Command]] = defaultdict(list)
        self._add_group((), root)

    def _add_group(self, path: tuple[str, ...], group: click.Group):
        assert path not in self.groups
        self.groups[path] = group

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

        self.groups[group].add_command(command)
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


def patch_pype():
    """
    revng2 is based on `pype`, but we want to change some defaults to be revng specific,
    and we want to add some commands.
    """

    # Replace the name (needed for autocompletion and usage)
    pype.name = "revng2"

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
    """Entry point for revng2."""
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(1))
    patch_pype()

    # Create and populate the registry
    registry = GroupRegistry(pype)
    load_commands(registry)
    registry.check()

    pype_main(sys.argv[1:], ContextObject)


if __name__ == "__main__":
    main()
