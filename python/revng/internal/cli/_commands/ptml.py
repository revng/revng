#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys
from contextlib import suppress
from io import TextIOWrapper
from typing import Callable, Optional, ParamSpec, TypeVar

from revng.ptml import PlainConsole, parse
from revng.ptml.common import log, normalize_filter_extract

from ..commands_registry import Command, CommandsRegistry, Options

T = TypeVar("T")
P = ParamSpec("P")


def suppress_brokenpipe(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Optional[T]:
    """When running a program with a pipe, BrokenPipeError might be raised. This signals that the
    output pipe was closed, which we expect. Suppressing the exception is not enough since it can
    also happen at shutdown, which will trigger python's unraisable hook, to remedy this we
    overwrite the default hook to ignore BrokenPipeError."""

    def new_unraisablehook(arg):
        if arg.exc_type != BrokenPipeError:
            sys.__unraisablehook__(arg)

    sys.unraisablehook = new_unraisablehook

    with suppress(BrokenPipeError):
        return func(*args)
    return None


def cmd_text(args):
    if args.inplace and args.input == sys.stdin.buffer:
        log("Cannot strip inplace while reading from stdin")
        return 1

    filters = normalize_filter_extract(args.filter, args.extract)
    content = args.input.read()

    if args.inplace:
        args.input.seek(0)
        args.input.truncate(0)
        output = TextIOWrapper(args.input, "utf-8")
    else:
        output = args.output

    if not args.color:
        console = PlainConsole(output)
    else:
        try:
            # rich takes a while to import, do it if needed
            from rich.console import Console

            console = Console(markup=False, highlight=False, force_terminal=True, file=output)
            console.options.no_wrap = True
            color = True
        except ImportError:
            console = PlainConsole(output)
            color = False

    if args.color and not color:
        log("Module 'rich' not found, please install it to use color mode")
        return 1

    return parse(content, console, filters)


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
