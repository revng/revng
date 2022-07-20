#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import re
import sys
from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from xml.dom import Node
from xml.dom.minidom import Document, parseString

import yaml

from .commands_registry import Command, Options, commands_registry

try:
    from rich.console import Console

    color = True
except ImportError:
    color = False


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


def yaml_load(content: str) -> Dict[str, Any]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(content, loader)


def yaml_dump(obj: Any) -> str:
    dumper = getattr(yaml, "CDumper", yaml.Dumper)
    return yaml.dump(obj, Dumper=dumper)


def normalize_filter_extract(filters: List[str], extract: Optional[str]) -> Union[str, List[str]]:
    if extract is not None:
        return extract
    if len(filters) == 0:
        return []
    return ",".join(filters).split(",")


def is_ptml(content: str) -> bool:
    return bool(re.match(r"\s*<", content))


T = TypeVar("T")


def handle_file(
    raw: str,
    func_one: Callable[[str], T],
    func_many: Callable[[Dict[str, str]], T],
    filters: Union[str, List[str]],
) -> T:
    if is_ptml(raw):
        if len(filters) > 0:
            print("Cannot extract/filter a plain ptml file", file=sys.stderr)
            sys.exit(1)

        return func_one(raw)
    else:
        parsed_yaml = yaml_load(raw)
        if isinstance(filters, str):
            if filters in parsed_yaml:
                return func_one(parsed_yaml[filters])
            else:
                return func_one("")
        else:
            filtered_yaml = {
                key: value
                for key, value in parsed_yaml.items()
                if len(filters) == 0 or key in filters
            }
            return func_many(filtered_yaml)


def strip_ptml(content: str) -> str:
    dom = parseString(content)
    result: List[str] = []
    _strip_ptml(dom, result)
    return "".join(result)


def _strip_ptml(node: Document, parts: List[str]):
    for node in node.childNodes:
        if node.nodeType == Node.TEXT_NODE:
            parts.append(node.nodeValue)
        elif node.nodeType == Node.ELEMENT_NODE:
            _strip_ptml(node, parts)
        else:
            pass


def strip_ptml_many(content: Dict[str, str]) -> str:
    result = {key: strip_ptml(value) for key, value in content.items()}
    return yaml_dump(result)


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
    if args.color and not color:
        print("Module 'rich' not found, please install it to use color mode")
        sys.exit(1)

    filters = normalize_filter_extract(args.filter, args.extract)
    content = args.input.read()
    if args.plain or not color:
        handle_file(content, cat_ptml_plain_single, cat_ptml_plain_many, filters)
    else:
        console = Console(markup=False, highlight=False, force_terminal=args.color)
        console.options.no_wrap = True
        handle_file(
            content,
            lambda x: cat_ptml_color_single(x, console),
            lambda x: cat_ptml_color_many(x, console),
            filters,
        )


def cmd_strip(args):
    if args.inplace and args.input == sys.stdin:
        print("Cannot strip inplace while reading from stdin", file=sys.stderr)
        sys.exit(1)

    filters = normalize_filter_extract(args.filter, args.extract)
    content = args.input.read()
    result = handle_file(content, strip_ptml, strip_ptml_many, filters)
    if args.inplace:
        args.input.seek(0)
        args.input.truncate(0)
        args.input.write(result)
    else:
        args.output.write(result)


class PTMLCommand(Command):
    def __init__(self):
        super().__init__(("ptml",), "Tool to manipulate PTML files", True)

    def register_arguments(self, parser):
        subparsers = parser.add_subparsers(title="Subcommands", metavar="")

        parser_cat = subparsers.add_parser("cat", help="Print the ptml on the console")
        parser_cat_filter = parser_cat.add_mutually_exclusive_group()
        parser_cat_filter.add_argument(
            "-f",
            "--filter",
            type=str,
            action="append",
            default=[],
            required=False,
            help="Only show the keys specified (if present)",
        )
        parser_cat_filter.add_argument(
            "-e", "--extract", type=str, required=False, help="Extract the specified key"
        )
        parser_cat_color = parser_cat.add_mutually_exclusive_group()
        parser_cat_color.add_argument(
            "-p", "--plain", action="store_true", help="Force plaintext output"
        )
        parser_cat_color.add_argument(
            "-c", "--color", action="store_true", help="Force color output"
        )
        parser_cat.add_argument(
            "input",
            type=argparse.FileType("r"),
            default=sys.stdin,
            nargs="?",
            help="Input file (stdin if omitted)",
        )
        parser_cat.set_defaults(func=cmd_cat)

        parser_strip = subparsers.add_parser("strip", help="Strip PTML from a file")
        parser_strip_filter = parser_strip.add_mutually_exclusive_group()
        parser_strip_filter.add_argument(
            "-f",
            "--filter",
            type=str,
            action="append",
            default=[],
            required=False,
            help="Only show the keys specified (if present)",
        )
        parser_strip_filter.add_argument(
            "-e", "--extract", type=str, required=False, help="Extract the specified key"
        )
        parser_strip.add_argument(
            "input",
            type=argparse.FileType("r+"),
            default=sys.stdin,
            nargs="?",
            help="Input file (stdin if omitted)",
        )
        parser_strip_out = parser_strip.add_mutually_exclusive_group()
        parser_strip_out.add_argument("-i", "--inplace", action="store_true", help="Strip inplace")
        parser_strip_out.add_argument(
            "output",
            type=argparse.FileType("w"),
            default=sys.stdout,
            nargs="?",
            help="Output file (stdout if omitted)",
        )
        parser_strip.set_defaults(func=cmd_strip)

    def run(self, options: Options):
        def new_unraisablehook(arg):
            if arg.exc_type != BrokenPipeError:
                sys.__unraisablehook__(arg)

        sys.unraisablehook = new_unraisablehook

        with suppress(BrokenPipeError, KeyboardInterrupt):
            options.parsed_args.func(options.parsed_args)


commands_registry.register_command(PTMLCommand())
