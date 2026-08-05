#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
import sys
import tarfile
from contextlib import contextmanager, suppress
from io import TextIOWrapper
from typing import IO, Any, Generator, List, Mapping, Optional, Union, cast

import click
import jsonschema
import yaml
from click_option_group import MutuallyExclusiveOptionGroup, optgroup

from revng.internal.cli.common import CommandRegistry
from revng.internal.cli.support import TarDictionary, file_wrapper
from revng.ptml.printer import ColorMode, ptml_print, ptml_print_mapping
from revng.support import log_error, to_bytes

# A PTML dictionary maps object ids to PTML documents. Since PTML is based on
# HTML, each document starts with the '<' of its root tag.
PTML_DICTIONARY_SCHEMA = {
    "type": "object",
    "propertyNames": {"type": "string"},
    "additionalProperties": {"type": "string", "pattern": r"^\s*<"},
}

PTML_DICTIONARY_VALIDATOR = jsonschema.Draft7Validator(PTML_DICTIONARY_SCHEMA)


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


def handler(
    input_: str | None,
    output: str | None,
    plain: bool,
    color_: bool,
    filter_: tuple[str, ...],
    extract: str | None,
    inplace: bool,
) -> int:
    if inplace and input_ in (None, "-"):
        log_error("Cannot strip inplace while reading from stdin")
        return 1

    filters = normalize_filter_extract(list(filter_), extract)

    color = ColorMode.Autodetect
    if color_:
        color = ColorMode.AutodetectForceColor
    if plain:
        color = ColorMode.Off

    if inplace:
        assert input_ is not None
        with open(input_, "rb+") as input_file:
            content = input_file.read()
            input_file.seek(0)
            input_file.truncate(0)
            return handler_inner(content, TextIOWrapper(input_file, "utf-8"), color, filters)
    else:
        with file_wrapper(input_, "rb") as input_file, file_wrapper(output, "w") as output_file:
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
            log_error("The input is empty")
            return 1

        # Try and read the input as a tar file with one or more PTML files
        with suppress(tarfile.ReadError):
            mapper = TarDictionary(wrapped)
            return handle_filters(mapper, filters, output, color)

        # PTML is based on HTML, so a '<' as the first non-whitespace character
        # means the input is a single PTML document.
        if re.match(rb"\s*<", wrapped) is not None:
            ptml_print(wrapped, output, color)
            return 0

        # Otherwise the input might be a YAML dictionary of PTML documents.
        data: Any = None
        with suppress(yaml.YAMLError):
            data = yaml.load(wrapped, Loader=yaml.CSafeLoader)
        if is_ptml_dictionary(data):
            return handle_filters(data, filters, output, color)

        log_error(
            "The input is in none of the formats this command reads: a PTML "
            "document, a YAML dictionary of PTML documents, or a tar archive "
            "of PTML documents"
        )
        return 1


def is_ptml_dictionary(data: Any) -> bool:
    """Checks that what YAML gave us is a dictionary of PTML documents. YAML is
    permissive enough that most inputs parse successfully (a plain text file,
    for instance, parses as a string), so a successful parse on its own does
    not mean the input really is a YAML file."""
    return PTML_DICTIONARY_VALIDATOR.is_valid(data)


def handle_filters(
    data: Mapping[str, bytes], filters: str | list[str], output: IO[str], color: ColorMode
) -> int:
    if isinstance(filters, str):
        if filters not in data:
            log_error(f"Could not find `{filters}` in the input")
            return 1
        ptml_print(data[filters], output, color)
    elif len(filters) == 0:
        ptml_print_mapping(data, output, color, lambda x: True)
    else:
        ptml_print_mapping(data, output, color, lambda x: x in cast(List[str], filters))
    return 0


@click.command(name="ptml", help="Tool to manipulate PTML files")
@click.argument("input_", metavar="[INPUT]", required=False)
@optgroup.group(
    "Output Format",
    cls=MutuallyExclusiveOptionGroup,
    help="Picks the output format, if omitted it will be color on terminal "
    "(if supported) or plain otherwise",
)
@optgroup.option("-p", "--plain", is_flag=True, help="Plaintext output")
@optgroup.option("-c", "--color", "color_", is_flag=True, help="Color output")
@optgroup.group("Output Filtering", cls=MutuallyExclusiveOptionGroup)
@optgroup.option(
    "-f",
    "--filter",
    "filter_",
    type=str,
    multiple=True,
    help="Only show the specified comma-separated keys (if present)",
)
@optgroup.option("-e", "--extract", type=str, help="Extract the specified key")
@optgroup.group("Output", cls=MutuallyExclusiveOptionGroup)
@optgroup.option("-i", "--inplace", is_flag=True, help="Strip inplace")
@optgroup.option("-o", "--output", metavar="FILE", help="Output file (stdout if omitted)")
def ptml(
    input_: str | None,
    plain: bool,
    color_: bool,
    filter_: tuple[str, ...],
    extract: str | None,
    inplace: bool,
    output: str | None,
) -> Optional[int]:
    with suppress(KeyboardInterrupt), suppress_brokenpipe():
        return handler(input_, output, plain, color_, filter_, extract, inplace)
    return 0


def setup(registry: CommandRegistry):
    registry.register((), ptml)
