#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import dataclasses
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .support import Options, collect_files, executable_name, try_run


class Command(ABC):
    def __init__(self, command: Tuple[str], help_text: str, add_help=True):
        self.namespace = command[:-1]
        self.name = command[-1]
        self.help = help_text
        self.add_help = add_help

    @abstractmethod
    def register_arguments(self, parser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def run(self, options: Options) -> Optional[int]:
        raise NotImplementedError("Please implement this method")


class ExternalCommand(Command):
    def __init__(self, command, path):
        super().__init__(command, f"see {executable_name()} {' '.join(command)} --help", False)
        self.path = path

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        return try_run([self.path] + options.remaining_args, options)


COMMAND_ARG_PREFIX = "command_name"


class CommandsRegistry:
    def __init__(self):
        parser = argparse.ArgumentParser(description="revng driver.")
        parser.add_argument("--version", action="store_true", help="Display version information.")
        parser.add_argument(
            "--dry-run", action="store_true", help="Do not execute any external command."
        )
        parser.add_argument("--verbose", action="store_true", help="Log all executed commands.")
        parser.add_argument(
            "--keep-temporaries", action="store_true", help="Do not delete temporary files."
        )
        parser.add_argument(
            "--perf", action="store_true", help="Run programs under perf (for use with hotspot)."
        )
        parser.add_argument(
            "--heaptrack", action="store_true", help="Run programs under heaptrack."
        )
        parser.add_argument("--gdb", action="store_true", help="Run programs under gdb.")
        parser.add_argument("--lldb", action="store_true", help="Run programs under lldb.")
        parser.add_argument("--valgrind", action="store_true", help="Run programs under valgrind.")
        parser.add_argument(
            "--callgrind", action="store_true", help="Run programs under callgrind."
        )
        parser.add_argument("--rr", action="store_true", help="Run programs under rr.")
        parser.add_argument(
            "--prefix", action="append", metavar="PREFIX", help="Additional search prefix."
        )
        self.root_parser = parser
        self.commands = {}
        self.namespaces = {}
        self.namespaces[()] = self.root_parser.add_subparsers(dest=f"{COMMAND_ARG_PREFIX}0")

    def register_command(self, command: Command):
        self.commands[command.namespace + (command.name,)] = command

    def has_command(self, command: str):
        return self._parse_command(command) in self.commands

    def register_external_command(self, command: str, path: str):
        self.register_command(ExternalCommand(self._parse_command(command), path))

    def run(self, arguments, options: Options):
        original_options = dict(options.__dict__.items())
        (args, rest) = self.root_parser.parse_known_args(arguments)
        command = []
        for name, value in args.__dict__.items():
            if name.startswith(COMMAND_ARG_PREFIX):
                index = int(name[len(COMMAND_ARG_PREFIX) :])
                command.append((index, value))

        command_sorted = tuple(value for _, value in sorted(command, key=lambda pair: pair[0]))
        if command_sorted[-1] is None:
            command_sorted = command_sorted[:-1]

        options = dataclasses.replace(options)
        options.parsed_args = args
        options.remaining_args = rest
        if not options.verbose:
            options.verbose = args.verbose
        if not options.dry_run:
            options.dry_run = args.dry_run
        if not options.keep_temporaries:
            options.keep_temporaries = args.keep_temporaries

        if len(options.command_prefix) == 0:
            assert (
                sum(
                    (args.perf, args.heaptrack, args.gdb, args.lldb, args.valgrind, args.callgrind)
                    + (args.rr,)
                )
                <= 1
            )

            if args.gdb:
                options.command_prefix += ["gdb", "-q", "--args"]

            if args.lldb:
                options.command_prefix += ["lldb", "--"]

            if args.valgrind:
                suppressions = collect_files(options.search_prefixes, ["share", "revng"], "*.supp")
                options.command_prefix += [
                    "valgrind",
                    *(f"--suppressions={s}" for s in suppressions),
                ]

            if args.callgrind:
                options.command_prefix += ["valgrind", "--tool=callgrind"]

            if args.rr:
                options.command_prefix += ["rr"]

            if args.perf:
                options.command_prefix += [
                    "perf",
                    "record",
                    "--call-graph",
                    "dwarf",
                    "--output=perf.data",
                ]

            if args.heaptrack:
                options.command_prefix += ["heaptrack"]

        if args.version:
            hashes = collect_files(
                options.search_prefixes, ["share", "revng", "component-hashes"], "*"
            )
            version = "-".join(Path(path).read_text().strip()[:7] for path in sorted(hashes))
            sys.stdout.write(f"rev.ng version {version}\n")
            return 0

        if command_sorted in self.commands:
            return self.commands[command_sorted].run(options)
        elif command_sorted in self.namespaces:
            return self.run((*arguments, "--help"), Options(**original_options))
        else:
            return 1

    def define_namespace(self, namespace: Sequence[str], help_text: Optional[str] = None):
        parent_parser = self._get_namespace_parser(namespace[:-1])
        new_parser = parent_parser.add_parser(
            namespace[-1], **({"help": help_text} if help_text is not None else {})
        )
        result = new_parser.add_subparsers(dest=f"{COMMAND_ARG_PREFIX}{len(namespace)}")
        self.namespaces[namespace] = result
        return result

    def _parse_command(self, command: str):
        parts = command.split("-")
        total = len(parts)
        current_namespace: Tuple[str, ...] = ()

        start_index = 0
        found = True
        while found and start_index < total:
            found = False
            for end_index in range(start_index + 1, total + 1):
                candidate = "-".join(parts[start_index:end_index])
                new_namespace = current_namespace + (candidate,)
                if new_namespace in self.namespaces:
                    current_namespace = new_namespace
                    start_index = end_index
                    found = True
                    break

        command_name = "-".join(parts[start_index:])

        return current_namespace + (command_name,)

    def post_init(self):
        for command in self.commands.values():
            parser = self._get_namespace_parser(command.namespace)
            command.register_arguments(
                parser.add_parser(command.name, help=command.help, add_help=command.add_help)
            )

    def _get_namespace_parser(self, namespace):
        if namespace not in self.namespaces:
            self.define_namespace(namespace)
        return self.namespaces[namespace]


commands_registry = CommandsRegistry()
commands_registry.define_namespace(
    ("model",), f"Model manipulation helpers, see {executable_name()} model --help"
)
commands_registry.define_namespace(("tar",), "Manipulate tar archives")
commands_registry.define_namespace(
    ("model", "import"), f"Model import helpers, see {executable_name()} model import --help"
)
commands_registry.define_namespace(
    ("trace",), f"Trace-related tools, see {executable_name()} trace --help"
)
