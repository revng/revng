#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
from io import TextIOWrapper
from typing import Any, Dict, List, Union
from xml.dom import Node
from xml.dom.minidom import Document, parseString

from revng.internal.cli.support import file_wrapper

from .common import handle_file, log, normalize_filter_extract

COLOR_CONVERSION = {
    "comment": "white",
    "asm.label": "bright_red",
    "asm.label-indicator": "bright_white",
    "asm.mnemonic": "dodger_blue1",
    "asm.mnemonic-prefix": "bright_green",
    "asm.mnemonic-suffix": "bright_green",
    "asm.immediate-value": "dark_slate_gray2",
    "asm.memory-operand": "dodger_blue1",
    "asm.register": "orange1",
    "asm.helper": "dodger_blue1",
    "asm.directive": "sea_green1",
    "asm.instruction-address": "medium_purple1",
    "asm.raw-bytes": "white",
    "c.function": "bright_red",
    "c.type": "sea_green1",
    "c.operator": "dodger_blue1",
    "c.function_parameter": "orange1",
    "c.variable": "orange1",
    "c.field": "orange1",
    "c.constant": "dark_slate_gray2",
    "c.string_literal": "medium_purple1",
    "c.keyword": "dodger_blue1",
    "c.directive": "dodger_blue1",
    "doxygen.keyword": "turquoise2",
    "doxygen.identifier": "tan",
}


class PlainConsole:
    def __init__(self, file: TextIOWrapper):
        self.file = file

    def print(self, text: str, *, end: str = "\n", **kwargs):  # noqa: A003
        self.file.write(f"{text}{end}")

    def __del__(self):
        self.file.flush()


def parse_ptml_plain(content: str, console):
    dom = parseString(content)
    _parse_ptml_node(dom, console, "", {})


def parse_ptml_yaml(content: Dict[str, str], console):
    for key, value in content.items():
        # Output the yaml key manually, this is for two main reasons:
        # * rich.Console works well with file object, having strings would be cumbersome
        # * pyyaml cannot output strings as flow scalars without some hacking around
        console.print(f"{key}:", style="yellow1", end="")
        console.print(" |-")
        dom = parseString(value)
        console.print("  ", end="")
        _parse_ptml_node(dom, console, "  ", {})
        console.print("\n", end="")


def _parse_ptml_node(node: Document, console, indent: str, metadata: Dict[str, str]):
    for node in node.childNodes:
        if node.nodeType == Node.TEXT_NODE:
            content = re.sub("\n", f"\n{indent}", node.nodeValue)
            if "data-token" in metadata and metadata["data-token"] in COLOR_CONVERSION:
                console.print(content, end="", style=COLOR_CONVERSION[metadata["data-token"]])
            else:
                console.print(content, end="")
        elif node.nodeType == Node.ELEMENT_NODE:
            new_metadata = {**metadata}
            for key, value in node.attributes.items():
                new_metadata[key] = value
            _parse_ptml_node(node, console, indent, new_metadata)


def cmd_text(args) -> int:
    if args.inplace and args.input in (None, "-"):
        log("Cannot strip inplace while reading from stdin")
        return 1

    filters = normalize_filter_extract(args.filter, args.extract)
    if args.inplace:
        with open(args.input, "rb+") as input_file:
            content = input_file.read()
            input_file.seek(0)
            input_file.truncate(0)
            return cmd_text_inner(args.color, content, TextIOWrapper(input_file, "utf-8"), filters)
    else:
        with file_wrapper(args.input, "rb") as input_file:
            content = input_file.read()
            with file_wrapper(args.output, "w") as output_file:
                return cmd_text_inner(args.color, content, output_file, filters)


def cmd_text_inner(
    color: bool, content: bytes, output: TextIOWrapper, filters: Union[str, List[str]]
):
    console: Any
    if not color:
        console = PlainConsole(output)
    else:
        try:
            # rich takes a while to import, do it if needed
            from rich.console import Console

            console = Console(markup=False, highlight=False, force_terminal=True, file=output)
            console.options.no_wrap = True  # type: ignore
            color_available = True
        except ImportError:
            console = PlainConsole(output)
            color_available = False

    if color and not color_available:
        log("Module 'rich' not found, please install it to use color mode")
        return 1

    return handle_file(
        content,
        lambda x: parse_ptml_plain(x, console),
        lambda x: parse_ptml_yaml(x, console),
        filters,
    )
