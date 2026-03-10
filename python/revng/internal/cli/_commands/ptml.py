#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import re
import sys
import tarfile
from contextlib import contextmanager, suppress
from io import TextIOWrapper
from typing import IO, Generator, List, Mapping, Optional, Union, cast

import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import TarDictionary, file_wrapper
from revng.ptml.printer import ColorMode, ptml_print, ptml_print_mapping
from revng.support import to_bytes


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
    content: bytes | IO[bytes], output: IO[str], color: ColorMode, filters: str | list[str]
):
    with to_bytes(content) as wrapped:
        # Here we have the raw bytes of input, we need to figure out if the
        # input file is a tar, a yaml or a plain PTML file. None of these
        # format have magic header bytes to identify them, so the only way to
        # read them is trying and move on if there is an exception.

        if len(wrapped) == 0:
            raise ValueError("Input is empty!")

        # Try and read the input as a tar file with one or more PTML files
        with suppress(tarfile.ReadError):
            mapper = TarDictionary(wrapped)
            handle_filters(mapper, filters, output, color)
            return 0

        # PTML is based on HTML, so we should be seeing a '<' as the first
        # non-whitespace character, if this is not the case it might be a YAML
        # file, try and read it as such.
        if re.match(rb"\s*<", wrapped) is None:
            data = None
            with suppress(yaml.YAMLError):
                data = yaml.load(wrapped, Loader=yaml.CSafeLoader)
            if data is not None:
                handle_filters(data, filters, output, color)
                return 0

        # We've tried all other options, try and read the file as a plain PTML
        ptml_print(wrapped, output, color)

    return 0


def handle_filters(
    data: Mapping[str, bytes], filters: str | list[str], output: IO[str], color: ColorMode
):
    if isinstance(filters, str):
        ptml_print(data[filters], output, color)
    elif len(filters) == 0:
        ptml_print_mapping(data, output, color, lambda x: True)
    else:
        ptml_print_mapping(data, output, color, lambda x: x in cast(List[str], filters))


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
