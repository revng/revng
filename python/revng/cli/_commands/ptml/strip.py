#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys
from typing import Dict

from revng.cli.commands_registry import Command, Options

from .common import (
    handle_file,
    log,
    normalize_filter_extract,
    strip_ptml,
    suppress_brokenpipe,
    yaml_dump,
)


def strip_ptml_many(content: Dict[str, str]) -> str:
    result = {key: strip_ptml(value) for key, value in content.items()}
    return yaml_dump(result)


def cmd_strip(args):
    if args.inplace and args.input == sys.stdin:
        log("Cannot strip inplace while reading from stdin")
        return 1

    filters = normalize_filter_extract(args.filter, args.extract)
    content = args.input.read()
    result = handle_file(content, strip_ptml, strip_ptml_many, filters)
    if args.inplace:
        args.input.seek(0)
        args.input.truncate(0)
        args.input.write(result)
    else:
        args.output.write(result)
    return 0


class PTMLStripCommand(Command):
    def __init__(self):
        super().__init__(("ptml", "strip"), "Strip PTML from a file")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser_filter = parser.add_mutually_exclusive_group()
        parser_filter.add_argument(
            "-f",
            "--filter",
            type=str,
            action="append",
            default=[],
            required=False,
            help="Only show the keys specified (if present)",
        )
        parser_filter.add_argument(
            "-e", "--extract", type=str, required=False, help="Extract the specified key"
        )
        parser.add_argument(
            "input",
            type=argparse.FileType("r+"),
            default=sys.stdin,
            nargs="?",
            help="Input file (stdin if omitted)",
        )
        parser_strip_out = parser.add_mutually_exclusive_group()
        parser_strip_out.add_argument("-i", "--inplace", action="store_true", help="Strip inplace")
        parser_strip_out.add_argument(
            "output",
            type=argparse.FileType("w"),
            default=sys.stdout,
            nargs="?",
            help="Output file (stdout if omitted)",
        )
        parser.set_defaults(func=cmd_strip)

    def run(self, options: Options) -> int:
        return suppress_brokenpipe(cmd_strip, options.parsed_args)
