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
    color: str = ""
    bgcolor: str = ""
    entries: List[str] = field(default_factory=list)

    def name(self) -> str:
        return f"node_{id(self)}"

    def to_graphviz(self) -> str:
        result = ""
        result += f"""  {escape(self.name())} ["""
        result += "label=<\n"
        result += """    <table border="0" cellborder="1" cellspacing="0">\n"""

        cell_properties = ""
        if self.bgcolor:
            cell_properties = f' bgcolor="{self.bgcolor}"'

        result += f"""      <tr><td{cell_properties}><b>{escape(self.label)}</b></td></tr>\n"""

        for index, entry in enumerate(self.entries):
            result += f"""      <tr><td port="entry_{index}">{escape(entry)}</td></tr>\n"""
        result += "    </table>\n"
        result += "  >"
        if self.color:
            result += f', color="{self.color}"'
        result += "];\n\n"
        return result

    def __hash__(self) -> int:
        return hash(self.name())


@dataclass(frozen=True, slots=True)
class Edge:
    source: Node
    destination: Node
    source_port: int = 0
    destination_port: int = 0
    head_label: str = ""
    tail_label: str = ""
    style: str = ""
    color: str = ""

    def to_graphviz(self) -> str:
        result = ""
        result += "  "
        if self.source_port >= 0:
            result += f"{escape(self.source.name())}:entry_{self.source_port}"
        else:
            result += f"{escape(self.source.name())}"

        result += " -> "

        if self.destination_port >= 0:
            result += f"{escape(self.destination.name())}:entry_{self.destination_port}"
        else:
            result += f"{escape(self.destination.name())}"

        attributes = []
        if self.head_label:
            attributes.append(f'headlabel="{escape(self.head_label)}"')
        if self.tail_label:
            attributes.append(f'taillabel="{escape(self.tail_label)}"')
        if self.style:
            attributes.append(f'style="{escape(self.style)}"')
        if self.color:
            attributes.append(f'color="{escape(self.color)}"')

        if len(attributes) > 0:
            result += f" [{','.join(attributes)}]"

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
