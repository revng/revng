#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options

from .log import set_verbose
from .merge_dynamic import merge_dynamic


class MergeDynamicCommand(Command):
    def __init__(self):
        super().__init__(
            ("merge-dynamic",),
            "Merge the dynamic portions of the translate ELF with the one from the host ELF",
        )

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "to_extend",
            metavar="TO_EXTEND",
            help="The ELF to extend.",
        )
        parser.add_argument(
            "source",
            metavar="SOURCE",
            help="The original ELF.",
        )
        parser.add_argument(
            "output",
            metavar="OUTPUT",
            nargs="?",
            type=argparse.FileType("wb"),
            default=sys.stdout,
            help="The output ELF (stdout if omitted)",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print debug information and warnings.",
        )
        parser.add_argument(
            "--base",
            metavar="ADDRESS",
            default="0x400000",
            help="The base address where dynamic object have been loaded.",
        )
        parser.add_argument(
            "--merge-load-segments",
            action="store_true",
            help="Merge the LOADed segments from the source ELF into the output ELF.",
        )

    def run(self, options: Options) -> int:
        args = options.parsed_args
        set_verbose(args.verbose)

        base = int(args.base, base=0)

        with open(args.source, "rb") as source_file, open(args.to_extend, "rb") as to_extend_file:
            retcode = merge_dynamic(
                to_extend_file,
                source_file,
                args.output,
                base=base,
                merge_load_segments=args.merge_load_segments,
            )

        args.output.flush()
        return retcode


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(MergeDynamicCommand())
