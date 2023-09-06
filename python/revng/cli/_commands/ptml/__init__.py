#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys
from typing import Optional

from revng.cli.commands_registry import Command, CommandsRegistry, Options

from .common import suppress_brokenpipe
from .text import cmd_text


class PTMLCommand(Command):
    def __init__(self):
        super().__init__(("ptml",), "Tool to manipulate PTML files")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.description = "Tool to manipulate PTML files"
        parser.add_argument(
            "input",
            type=argparse.FileType("rb+"),
            default=sys.stdin.buffer,
            nargs="?",
            help="Input file (stdin if omitted)",
        )

        parser_format_group = parser.add_argument_group(
            "Output Format",
            "Picks the output format, if omitted it will be color on terminal\n"
            "(if supported) or plain otherwise",
        )
        parser_format = parser_format_group.add_mutually_exclusive_group()
        parser_format.add_argument("-p", "--plain", action="store_true", help="Plaintext output")
        parser_format.add_argument(
            "-c", "--color", action="store_true", help="Color output (requires the rich package)"
        )

        parser_filter_group = parser.add_argument_group("Output Filtering")
        parser_filter = parser_filter_group.add_mutually_exclusive_group()
        parser_filter.add_argument(
            "-f",
            "--filter",
            type=str,
            action="append",
            default=[],
            required=False,
            help="Only show the specified comma-separated keys (if present)",
        )
        parser_filter.add_argument(
            "-e", "--extract", type=str, required=False, help="Extract the specified key"
        )

        parser_out_group = parser.add_argument_group("Output")
        parser_out = parser_out_group.add_mutually_exclusive_group()
        parser_out.add_argument("-i", "--inplace", action="store_true", help="Strip inplace")
        parser_out.add_argument(
            "-o",
            "--output",
            type=argparse.FileType("w"),
            default=sys.stdout,
            nargs="?",
            metavar="FILE",
            help="Output file (stdout if omitted)",
        )

    def run(self, options: Options) -> Optional[int]:
        args = options.parsed_args
        return suppress_brokenpipe(cmd_text, args)


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(PTMLCommand())
