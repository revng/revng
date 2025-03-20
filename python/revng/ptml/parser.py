#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# noqa: A003

from __future__ import annotations

import xml.sax
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Generator, List

from revng.support import ToBytesInput, to_bytes


@dataclass
class Position:
    """Represents a single character in a multi-line text document"""

    line: int
    character: int

    def clone(self):
        return Position(self.line, self.character)

    def to_string(self):
        return f"{self.line}:{self.character}"


@dataclass
class Range:
    """Represents a range of characters in a multi-line text document. Both
    start and end are inclusive"""

    start: Position
    end: Position

    def is_single_line(self):
        return self.start.line == self.end.line

    def to_string(self):
        return f"{self.start.to_string()},{self.end.to_string()}"


class TextExtractor:
    """Utility function that allows extracting `Range`s from a text"""

    def __init__(self, text: str):
        # This is stored just so the lines property can be computed lazily via
        # the cached_property decorator
        self.original_text = text

    @cached_property
    def lines(self) -> List[str]:
        return self.original_text.split("\n")

    def get_text(self, range_: Range) -> str:
        if range_.is_single_line():
            return self._get_line(range_.start.line, range_.start.character, range_.end.character)
        lines = [self._get_line(range_.start.line, range_.start.character)]
        for line_index in range(range_.start.line, range_.end.line + 1):
            lines.append(self.lines[line_index])
        lines.append(self._get_line(range_.end.line, -1, range_.end.character))
        return "\n".join(lines)

    def _get_line(self, index: int, start=-1, end=-1):
        line = self.lines[index]
        if start == -1 and end == -1:
            return line
        elif start == -1:
            return line[:end]
        elif end == -1:
            return line[start:]
        else:
            return line[start:end]


@dataclass
class TreeNode:
    """The node of a tree of ranged attributes"""

    attributes: Dict[str, str]
    children: List[TreeNode]
    text_range: Range


@dataclass
class RangedAttributes:
    attributes: Dict[str, str]
    text_range: Range


class MetadataTree:
    """Utility class that allows interacting with a TreeNode tree more easily"""

    def __init__(self, root: TreeNode):
        self.root = root

    def walk(self) -> Generator[RangedAttributes, None, None]:
        queue = [self.root]
        index = 0
        while index < len(queue):
            top = queue[index]
            queue.extend(top.children)
            yield RangedAttributes(top.attributes, top.text_range)
            index += 1


class Parser(xml.sax.ContentHandler):
    """This ContentHandler will split the document into its bare text and a
    tree of TreeNodes stored in root"""

    def __init__(self):
        # The bare text
        self.text = ""
        # The root node of the document
        self.root = TreeNode({}, [], Range(Position(0, 0), Position(0, 0)))
        self._currentPosition = Position(0, 0)
        self._stack = [self.root]

    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesImpl):  # noqa: N802
        assert name in ("div", "span")
        new_node = TreeNode(dict(attrs), [], Range(self._currentPosition.clone(), Position(0, 0)))
        self._stack[-1].children.append(new_node)
        self._stack.append(new_node)

    def endElement(self, name: str):  # noqa: N802
        # Mark the top-most node's end position
        self._stack[-1].text_range.end = self._currentPosition.clone()
        # Pop the top-most node from the stack
        self._stack.pop()

    def endDocument(self):  # noqa: N802
        # Check that we didn't miss closing an element
        assert len(self._stack) == 1 and self._stack[0] is self.root
        # Mark the end of the root node
        self.root.text_range.end = self._currentPosition.clone()

    def characters(self, content: str):
        self.text += content
        parts = content.split("\n")
        if len(parts) == 1:
            self._currentPosition.character += len(content)
        else:
            self._currentPosition.line += len(parts) - 1
            self._currentPosition.character = len(parts[-1])


@dataclass(frozen=True)
class PTMLDocument:
    text: str
    metadata: MetadataTree
    extractor: TextExtractor


def parse(input_: ToBytesInput) -> PTMLDocument:
    """Split a single PTML document into its text and metadata"""
    handler = Parser()
    with to_bytes(input_) as wrapped:
        xml.sax.parseString(wrapped, handler)
    return PTMLDocument(handler.text, MetadataTree(handler.root), TextExtractor(handler.text))
