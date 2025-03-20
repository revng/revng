#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys
from contextlib import contextmanager, suppress
from io import TextIOWrapper
from typing import IO, Generator, List, Optional, Union, cast

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import file_wrapper
from revng.ptml.printer import ColorMode
from revng.support import to_bytes
from revng.support.artifacts import PTMLArtifact, ptml_artifact_autodetect


def normalize_filter_extract(filters: List[str], extract: Optional[str]) -> Union[str, List[str]]:
    if extract is not None:
        return extract
    if len(filters) == 0:
        return []
    return ",".join(filters).split(",")


@contextmanager
def suppress_brokenpipe() -> Generator[None, None, None]:
    """When running a program with a pipe, BrokenPipeError might be raised. This signals that the
    output pipe was closed, which we expect. Suppressing the exception is not enough since it can
    also happen at shutdown, which will trigger python's unraisable hook, to remedy this we
    overwrite the default hook to ignore BrokenPipeError."""

    def new_unraisablehook(arg):
        if arg.exc_type != BrokenPipeError:
            sys.__unraisablehook__(arg)

    old_unraisablehook = sys.unraisablehook
    sys.unraisablehook = new_unraisablehook
    with suppress(BrokenPipeError):
        yield None
    sys.unraisablehook = old_unraisablehook


def handler(args) -> int:
    if args.inplace and args.input in (None, "-"):
        sys.stderr.write("Cannot strip inplace while reading from stdin\n")
        return 1

    filters = normalize_filter_extract(args.filter, args.extract)

    color = ColorMode.Autodetect
    if args.color:
        color = ColorMode.AutodetectForceColor
    if args.plain:
        color = ColorMode.Off

    if args.inplace:
        with open(args.input, "rb+") as input_file:
            content = input_file.read()
            input_file.seek(0)
            input_file.truncate(0)
            return handler_inner(content, TextIOWrapper(input_file, "utf-8"), color, filters)
    else:
        with file_wrapper(args.input, "rb") as input_file, file_wrapper(
            args.output, "w"
        ) as output_file:
            return handler_inner(input_file, output_file, color, filters)


def handler_inner(
    content: bytes | IO[bytes], output: IO[str], color: ColorMode, filters: Union[str, List[str]]
):
    with to_bytes(content) as wrapped:
        artifact = ptml_artifact_autodetect(wrapped)
        if isinstance(artifact, PTMLArtifact):
            artifact.print(output, color)
            return 0

        if isinstance(filters, str):
            artifact[filters].print(output, color)
        elif len(filters) == 0:
            artifact.print(output, color)
        else:
            artifact.print(output, color, lambda x: x in cast(List[str], filters))

    return 0


class PTMLCommand(Command):
    def __init__(self):
        super().__init__(("ptml",), "Tool to manipulate PTML files")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.description = "Tool to manipulate PTML files"
        parser.add_argument("input", nargs="?", help="Input file (stdin if omitted)")

        parser_format_group = parser.add_argument_group(
            "Output Format",
            "Picks the output format, if omitted it will be color on terminal\n"
            "(if supported) or plain otherwise",
        )
        parser_format = parser_format_group.add_mutually_exclusive_group()
        parser_format.add_argument("-p", "--plain", action="store_true", help="Plaintext output")
        parser_format.add_argument("-c", "--color", action="store_true", help="Color output")

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
            "-o", "--output", nargs="?", metavar="FILE", help="Output file (stdout if omitted)"
        )

    def run(self, options: Options) -> Optional[int]:
        with suppress(KeyboardInterrupt), suppress_brokenpipe():
            return handler(options.parsed_args)
        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(PTMLCommand())
