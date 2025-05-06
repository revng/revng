#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import re
import sys
import xml.sax
from enum import Enum
from typing import IO, Callable, Dict, List, Optional, Protocol, TextIO, cast

import yachalk
from yachalk.supports_color import detect_color_support

from revng.support import ToBytesInput, to_bytes


class PrinterBackend(Protocol):
    def print(self, content: str, attribute_getter: Callable[[str], Optional[str]]):  # noqa: A003
        """Printing implementation.
        * content is the text to print
        * attribute_getter is a getter function for the attributes in the
            current context
        """
        ...


class Printer(xml.sax.ContentHandler):
    """This ContentHandler will print the contents of the document while
    parsing, using the provided Backend instance. This is more efficient than
    using a Parser because the text is not saved and the metadata tree is
    never created."""

    def __init__(self, backend: PrinterBackend):
        self.attributes_stack: List[Dict[str, str]] = []
        self.backend = backend

    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesImpl):  # noqa: N802
        self.attributes_stack.append(dict(attrs))

    def endElement(self, name: str):  # noqa: N802
        self.attributes_stack.pop()

    def characters(self, content: str):
        self.backend.print(content, self.get_attributes)

    def get_attributes(self, key: str) -> Optional[str]:
        if len(self.attributes_stack) == 0:
            return None

        for i in range(len(self.attributes_stack) - 1, -1, -1):
            if key in self.attributes_stack[i]:
                return self.attributes_stack[i][key]
        return None


class ColorMode(Enum):
    # Autodetect, but fall back to Color16 at worst
    AutodetectForceColor = -2
    # Autodetect based on current environment variables
    Autodetect = -1
    # Don't use color
    Off = 0
    # Force 4-bit color palette
    Color16 = 1
    # Force 8-bit color palette
    Color256 = 2
    # Force 24-bit color palette (truecolor)
    TrueColor = 3


def _detect_color_support(output) -> yachalk.ColorMode:
    ci_env = None
    try:
        # Remove the `CI` env since it triggers a different codepath in
        # `detect_color_support`
        if "CI" in os.environ:
            ci_env = os.environ.pop("CI")
        return detect_color_support(cast(TextIO, output))
    finally:
        if ci_env is not None:
            os.environ["CI"] = ci_env


def _get_chalk(mode: ColorMode, output: IO[str]) -> yachalk.ChalkFactory:
    if mode == ColorMode.Autodetect:
        color_mode = _detect_color_support(output)
    elif mode == ColorMode.AutodetectForceColor:

        class FakeStdout:
            # Fake stdout to force autodetection
            def isatty(self):
                return True

        color_mode = _detect_color_support(FakeStdout())
        if color_mode == yachalk.ColorMode.AllOff:
            color_mode = yachalk.ColorMode.Basic16
    else:
        color_mode = yachalk.ColorMode(mode.value)
    return yachalk.ChalkFactory(color_mode)


class ColorPrinter(PrinterBackend):
    """PrinterBackend which displays content using ANSI escape codes"""

    def __init__(
        self, output: IO[str], *, indent: str = "", color: ColorMode = ColorMode.Autodetect
    ):
        self.output = output
        self.chalk = _get_chalk(color, output)
        self.color_table = self._get_color_table()
        self.indent = indent

    def _get_color_table(self) -> Dict[str, Callable[[str], str]]:
        return {
            "comment": self.chalk.hex("#c0c0c0"),
            "asm.label": self.chalk.hex("#ff0000"),
            "asm.label-indicator": self.chalk.hex("#ffffff"),
            "asm.mnemonic": self.chalk.hex("#0087ff"),
            "asm.mnemonic-prefix": self.chalk.hex("#00ff00"),
            "asm.mnemonic-suffix": self.chalk.hex("#00ff00"),
            "asm.immediate-value": self.chalk.hex("#5fffff"),
            "asm.memory-operand": self.chalk.hex("#0087ff"),
            "asm.register": self.chalk.hex("#ffaf00"),
            "asm.helper": self.chalk.hex("#0087ff"),
            "asm.directive": self.chalk.hex("#5fffaf"),
            "asm.instruction-address": self.chalk.hex("#af87ff"),
            "asm.raw-bytes": self.chalk.hex("#c0c0c0"),
            "c.function": self.chalk.hex("#ff0000"),
            "c.type": self.chalk.hex("#5fffaf"),
            "c.operator": self.chalk.hex("#0087ff"),
            "c.function_parameter": self.chalk.hex("#ffaf00"),
            "c.variable": self.chalk.hex("#ffaf00"),
            "c.field": self.chalk.hex("#ffaf00"),
            "c.constant": self.chalk.hex("#5fffff"),
            "c.string_literal": self.chalk.hex("#af87ff"),
            "c.keyword": self.chalk.hex("#0087ff"),
            "c.directive": self.chalk.hex("#0087ff"),
            "doxygen.keyword": self.chalk.hex("#00d7ff"),
            "doxygen.identifier": self.chalk.hex("#d7af87"),
        }

    def key_color(self):
        return self.chalk.hex("#ffff00")

    def print(self, content: str, attribute_getter: Callable[[str], Optional[str]]):  # noqa: A003
        output_text = re.sub("\n", f"\n{self.indent}", content)
        data_token = attribute_getter("data-token")
        if data_token is not None and data_token in self.color_table:
            self.output.write(self.color_table[data_token](output_text))
        else:
            self.output.write(output_text)


class PlainPrinter(PrinterBackend):
    """PrinterBackend that just prints the text as-is. It's faster than
    ColorPrinter(..., color=ColorMode.None) since no color lookup is made."""

    def __init__(self, output: IO[str], indent: str = ""):
        self.output = output
        self.indent = ""

    def print(self, content: str, attribute_getter: Callable[[str], Optional[str]]):  # noqa: A003
        output_text = re.sub("\n", f"\n{self.indent}", content)
        self.output.write(output_text)


def ptml_print_with_printer(input_: ToBytesInput, printer: PrinterBackend):
    """Given a PTML document, print its contents with the given printer"""
    with to_bytes(input_) as wrapped:
        xml.sax.parseString(wrapped, Printer(printer))


def ptml_print(
    input_: ToBytesInput, output: IO[str] = sys.stdout, color: ColorMode = ColorMode.Autodetect
):
    """Print the input PTML document to the specified output"""
    printer: PrinterBackend
    if color == ColorMode.Off:
        printer = PlainPrinter(output)
    else:
        printer = ColorPrinter(output, color=color)
    ptml_print_with_printer(input_, printer)
