#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import re
import sys
from typing import Dict
from xml.dom import Node
from xml.dom.minidom import Document, parseString

from revng.cli.commands_registry import Command, Options

from .common import handle_file, normalize_filter_extract, strip_ptml, suppress_brokenpipe

COLOR_CONVERSION = {
    "asm.label": "bright_red",
    "asm.label-indicator": "",
    "asm.comment-indicator": "white",
    "asm.mnemonic": "dodger_blue1",
    "asm.mnemonic-prefix": "bright_green",
    "asm.mnemonic-suffix": "bright_green",
    "asm.immediate-value": "dark_slate_gray2",
    "asm.memory-operand": "dodger_blue1",
    "asm.register": "orange1",
}


def cat_ptml_plain_single(content: str):
    print(strip_ptml(content))


def cat_ptml_plain_many(content: Dict[str, str]):
    for key, value in content.items():
        print(f"{key}:")
        print(re.sub("\n", "\n  ", f"  {strip_ptml(value)}"))


def cat_ptml_color_single(content: str, console):
    dom = parseString(content)
    _cat_ptml_color(dom, console, "", {})


def cat_ptml_color_many(content: Dict[str, str], console):
    for key, value in content.items():
        console.print(f"{key}:", style="yellow1")
        dom = parseString(value)
        console.print("  ", end="")
        _cat_ptml_color(dom, console, "  ", {})
        console.print("\n", end="")


def _cat_ptml_color(node: Document, console, indent: str, metadata: Dict[str, str]):
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
            _cat_ptml_color(node, console, indent, new_metadata)
        else:
            pass


def cmd_cat(args):
    filters = normalize_filter_extract(args.filter, args.extract)
    content = args.input.read()
    if args.plain:
        handle_file(content, cat_ptml_plain_single, cat_ptml_plain_many, filters)
        return 0

    try:
        from rich.console import Console

        color = True
    except ImportError:
        color = False

    if args.color and not color:
        print("Module 'rich' not found, please install it to use color mode")
        sys.exit(1)

    if color:
        console = Console(markup=False, highlight=False, force_terminal=args.color)
        console.options.no_wrap = True
        handle_file(
            content,
            lambda x: cat_ptml_color_single(x, console),
            lambda x: cat_ptml_color_many(x, console),
            filters,
        )
    else:
        handle_file(content, cat_ptml_plain_single, cat_ptml_plain_many, filters)
    return 0


class PTMLCatCommand(Command):
    def __init__(self):
        super().__init__(("ptml", "cat"), "Print the ptml on the console")

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
        parser_color = parser.add_mutually_exclusive_group()
        parser_color.add_argument(
            "-p", "--plain", action="store_true", help="Force plaintext output"
        )
        parser_color.add_argument("-c", "--color", action="store_true", help="Force color output")
        parser.add_argument(
            "input",
            type=argparse.FileType("r"),
            default=sys.stdin,
            nargs="?",
            help="Input file (stdin if omitted)",
        )
        parser.set_defaults(func=cmd_cat)

    def run(self, options: Options) -> int:
        return suppress_brokenpipe(cmd_cat, options.parsed_args)
