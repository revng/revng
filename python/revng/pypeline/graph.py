#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass, field
from typing import List, Set, TypeAlias


def escape(string: str) -> str:
    return string.replace("\\", "\\\\").replace('"', '\\"')


@dataclass(slots=True)
class Node:
    label: str
    completed: bool = False
    entries: List[str] = field(default_factory=list)

    def name(self) -> str:
        return f"node_{id(self)}"

    def to_graphviz(self) -> str:
        result = ""
        result += f"""  {escape(self.name())} ["""
        result += "label=<\n"
        result += """    <table border="0" cellborder="1" cellspacing="0">\n"""

        cell_properties = ""
        if self.completed:
            cell_properties = ' bgcolor="lightgreen"'

        result += f"""      <tr><td{cell_properties}><b>{escape(self.label)}</b></td></tr>\n"""

        for index, entry in enumerate(self.entries):
            result += f"""      <tr><td port="entry_{index}">{escape(entry)}</td></tr>\n"""
        result += "    </table>\n"
        result += "  >];\n"
        result += "\n"
        return result

    def __hash__(self) -> int:
        return hash(self.name())


@dataclass(slots=True)
class Edge:
    source: Node
    destination: Node
    source_port: int = 0
    destination_port: int = 0
    head_label: str = ""
    tail_label: str = ""

    def to_graphviz(self) -> str:
        result = ""
        result += "  "
        result += f"{escape(self.source.name())}:entry_{self.source_port}"
        result += " -> "
        result += f"{escape(self.destination.name())}:entry_{self.destination_port}"
        if self.head_label or self.tail_label:
            result += " ["
            if self.head_label:
                result += f'headlabel="{escape(self.head_label)}"'
            if self.tail_label:
                if self.head_label:
                    result += ","
                result += f'taillabel="{escape(self.tail_label)}"'
            result += "]"
        result += ";\n"
        return result


class Graph:
    """A graph data structure for rendering purposes."""

    Node: TypeAlias = Node
    Edge: TypeAlias = Edge

    def __init__(self) -> None:
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()

    def is_tree(self) -> bool:
        # TODO: implement
        return True

    def to_graphviz(self) -> str:
        result = "digraph structs {\n"
        result += "rankdir = LR;\n"
        result += "node [shape=plaintext];\n"
        result += "\n\n"
        for node in self.nodes:
            result += node.to_graphviz()
        for edge in self.edges:
            result += edge.to_graphviz()
        result += "\n}\n"
        return result
