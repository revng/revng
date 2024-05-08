#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

# This command compares two YAML files. In general, the reference YAML file is
# supposed to be included in the other one. For instance, {"a": 1, "b": 2}
# contains {"b": 2}. This is useful for enforcing the content of certain parts
# of a YAML file, in particular during testing.

# TODO: we need to have more reliable way to detect and dereference references
#       and ignore "fragile" keys.
# TODO: we need to use an optional schema to know what lists are actually sets

import argparse
import re
import sys
from collections import defaultdict

import yaml
from grandiso import find_motifs
from networkx import DiGraph
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options

args = None

fragile_keys = {"ID"}


def is_reference(string):
    return string.startswith("/Types/")


def dereference(root, string):
    assert is_reference(string)
    match = re.match(r"/Types/([^-]*)-", string)

    if match:
        type_id = int(match.groups()[0])
        for item in root["Types"]:
            if item["ID"] == type_id:
                return item

    return None


class Tag:
    def __init__(self, tag):
        self.tag = tag

    def __eq__(self, other):
        return self.tag == other.tag

    def __str__(self):
        return f"tag: {self.tag}"


class YAMLGraph:
    def __init__(self, root):
        self.graph = DiGraph()
        self.node_map = {}
        self.root = root
        self.visit_object(root)
        self.node_colors = {}

    def add_node(self, object_):
        object_id = id(object_)
        self.node_map[object_id] = object_
        self.graph.add_node(object_id)

    def add_edge(self, source_id, destination_id, label):
        label_object = Tag(label)
        label_id = id(label_object)
        self.add_node(label_object)
        self.graph.add_edge(source_id, label_id)
        self.graph.add_edge(label_id, destination_id)

    def visit_object(self, object_):
        object_type = type(object_)
        object_id = id(object_)

        if object_id in self.node_map:
            return

        self.add_node(object_)

        if object_type is list:
            for index, item in enumerate(object_):
                self.visit_object(item)
                if args.exact:
                    self.add_edge(object_id, id(item), index)
                else:
                    self.add_edge(object_id, id(item), "")
        elif object_type is dict:
            for key, value in object_.items():
                key = key.lstrip("$")
                value_type = type(value)
                if value_type is dict or value_type is list:
                    self.visit_object(value)
                    self.add_edge(object_id, id(value), key)
                if value_type is str and is_reference(value):
                    item = dereference(self.root, value)
                    if item is not None:
                        self.visit_object(item)
                        self.add_edge(object_id, id(item), key)

    @staticmethod
    def filter(object_):  # noqa: A003
        assert isinstance(object_, dict)
        filtered_dict = dict(object_)
        to_remove = []
        references = []
        for key, value in filtered_dict.items():
            value_type = type(value)
            if key in fragile_keys or (value_type is list or value_type is dict):
                to_remove.append(key)

            if value_type is str and is_reference(value):
                references.append(key)

        for key in to_remove:
            del filtered_dict[key]

        for key in references:
            filtered_dict[key] = "reference"

        return filtered_dict

    @staticmethod
    def get_label(object_):
        object_type = type(object_)
        if object_type is list:
            return f"List[{len(object_)}]"
        if object_type is Tag:
            return str(object_.tag)
        if object_type is dict:
            return yaml.dump(YAMLGraph.filter(object_))
        return str(object_)

    @staticmethod
    def escape(data):
        return data.replace('"', '\\"').replace("\n", "\\l")

    def write(self, path):
        with open(path, "w", encoding="utf-8") as output_file:
            # Emit header
            output_file.write("digraph {\n")
            output_file.write("  node [shape=box];\n")

            # Emit nodes
            for object_id in self.graph.nodes:
                object_ = self.node_map[object_id]
                if isinstance(object_, Tag):
                    source_id = list(self.graph.predecessors(object_id))[0]
                    destination_id = list(self.graph.successors(object_id))[0]
                    label = self.escape(self.get_label(object_.tag))
                    output_file.write(f"""  n{source_id} ->""")
                    output_file.write(f"""  n{destination_id}""")
                    output_file.write(f""" [label="{label}"];\n""")
                else:
                    extra = ""
                    if object_id in self.node_colors:
                        extra += ",style=filled"
                        extra += f',fillcolor="{self.node_colors[object_id]}"'
                    label = self.escape(self.get_label(object_))
                    output_file.write(f"""  n{object_id}""")
                    output_file.write(f""" [label="{label}"{extra}];\n""")

            # Emit footer
            output_file.write("}\n")

    def color_match(self, other, match):
        colors = [
            "#ff5f7e",
            "#4f7d9d",
            "#8873b3",
            "#d273b7",
            "#f672a9",
            "#fb7f8c",
            "#ff7c43",
            "#ffa600",
        ]

        for index, (reference_id, input_id) in enumerate(match.items()):
            color = colors[index % len(colors)]
            self.node_colors[reference_id] = color
            other.node_colors[input_id] = color

    def semantic_feasibility(self, other, exact, input_id, reference_id):
        input_object = other.node_map[input_id]
        input_type = type(input_object)

        reference_object = self.node_map[reference_id]
        reference_type = type(reference_object)

        if reference_type is not input_type:
            return False

        if input_type is dict:
            for key, value in list(reference_object.items()):
                if key.startswith("$") and (type(value) is list or type(value) is dict):
                    key = key[1:]
                    if key not in input_object or len(value) != len(input_object[key]):
                        return False

            reference_object = YAMLGraph.filter(reference_object)
            input_object = YAMLGraph.filter(input_object)

            if exact:
                return input_object == reference_object

            for key, value in reference_object.items():
                if key.startswith("-"):
                    if key[1:] in input_object:
                        # The key starts with `-` but it's present in
                        # input_object
                        return False
                elif key not in input_object or input_object[key] != value:
                    # Object entries do not match!
                    return False

            return True

        if input_type is list:
            return True

        return input_object == reference_object

    def is_subgraph(self, other, color=None):
        return self.compare(other, False, color)

    def is_equal(self, other, color=None):
        return self.compare(other, True, color)

    def compare(self, other, exact, color=None):
        def attr_match(motif_node_id: str, host_node_id: str, _motif, _host) -> bool:
            return self.semantic_feasibility(other, exact, host_node_id, motif_node_id)

        # Compute (un)interestingness
        interestingness = defaultdict(lambda: 2)
        shortest_paths = dict(all_pairs_shortest_path_length(self.graph))
        for node_id in self.graph:
            # TODO: here we are hardcoding `CustomName`, we need to take
            #       the time to write a small function computing how
            #       much disambiguation power a certain reference node
            #       has based on how many compatible nodes are there in
            #       the input graph, just looking at the semantic.
            has_customname = (
                isinstance(self.node_map[node_id], dict) and "CustomName" in self.node_map[node_id]
            )
            if self.graph.in_degree(node_id) == 0:
                # No incoming edges, it's the entry node
                interestingness[node_id] = 1
            elif has_customname:
                # Those with CustomName take priority
                interestingness[node_id] = 2

                # Distribute points depending on the distance
                for source, data in shortest_paths.items():
                    if source == node_id:
                        for destination, distance in data.items():
                            interestingness[destination] += distance

        # Rescale all values between 0 and 1 and turn uninterestigness into
        # interestingness
        max_value = max(interestingness.values())
        for node_id in interestingness:
            interestingness[node_id] = interestingness[node_id] / max_value

        result = find_motifs(
            self.graph,
            other.graph,
            limit=2,
            interestingness=interestingness,
            isomorphisms_only=exact,
            is_node_attr_match=attr_match,
        )

        if result and color:
            self.color_match(other, result[0])

        return len(result) > 0


def selftest():
    # Test --exact
    args.exact = True

    def test_equal(input_, reference):
        return YAMLGraph(reference).is_equal(YAMLGraph(input_))

    assert test_equal({}, {})
    assert test_equal(3, 3)
    assert not test_equal(2, 3)
    assert test_equal([3], [3])
    assert not test_equal([2], [3])
    assert not test_equal([], {})
    assert test_equal({"a": 2}, {"a": 2})
    assert not test_equal({"a": 2}, {})
    assert not test_equal({"a": 2, "b": 3}, {"b": 2, "a": 3})
    assert test_equal([1, 2, 3], [1, 2, 3])
    assert not test_equal([1, 3, 2], [1, 2, 3])

    # Test approximate for inclusion
    args.exact = False

    def test_subgraph(input_, reference):
        return YAMLGraph(reference).is_subgraph(YAMLGraph(input_))

    assert test_subgraph({}, {})
    assert not test_subgraph({}, {"a": 2})
    assert test_subgraph([3], [3])
    assert not test_subgraph([2], [3])
    assert not test_subgraph([], {})
    assert test_subgraph({"a": 2}, {"a": 2})
    assert test_subgraph({"a": 2}, {})
    assert test_subgraph({"a": 2, "b": 3}, {"b": 3, "a": 2})
    assert test_subgraph([1, 2, 3], [1, 2, 3])
    assert test_subgraph([1, 3, 2], [1, 2, 3])

    # Test references
    reference = {"a": "/Types/1-Type", "Types": [{"ID": 1, "b": 3}]}
    assert test_subgraph(
        {"a": "/Types/2-Type", "Types": [{"ID": 1, "b": 5}, {"ID": 2, "b": 3}]}, reference
    )
    assert not test_subgraph(
        {"a": "/Types/2-Type", "Types": [{"ID": 1, "b": 5}, {"ID": 2, "b": 4}]}, reference
    )

    return 0


def open_argument(path):
    if path == "-":
        return sys.stdin
    return open(path, encoding="utf-8")  # noqa: SIM115


class ModelCompareCommand(Command):
    def __init__(self):
        super().__init__(("model", "compare"), "Compare a YAML file against a reference")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("reference", metavar="REFERENCE", help="The reference file.")
        parser.add_argument(
            "input", metavar="INPUT", default="-", nargs="?", help="The input file."
        )
        parser.add_argument(
            "--exact",
            action="store_true",
            help=("Match exactly, containing the reference is not enough."),
        )
        parser.add_argument("--not", action="store_true", help="If it matches, return an error.")
        parser.add_argument(
            "--dump-graphs", action="store_true", help="Dump INPUT.dot and REFERENCE.dot."
        )
        parser.add_argument("--selftest", action="store_true", help="Run internal tests.")

    def run(self, options: Options) -> int:
        global args
        args = options.parsed_args

        if args.selftest:
            return selftest()

        with open_argument(args.reference) as reference_file, open_argument(
            args.input
        ) as input_file:
            reference = yaml.safe_load(reference_file)
            input_ = yaml.safe_load(input_file)

        reference_graph = YAMLGraph(reference)
        input_graph = YAMLGraph(input_)

        if args.exact:
            result = reference_graph.is_equal(input_graph, args.dump_graphs)
        else:
            result = reference_graph.is_subgraph(input_graph, args.dump_graphs)

        if args.dump_graphs:
            reference_graph.write(f"{args.reference}.dot")
            input_graph.write(f"{args.input}.dot")

        if args.__dict__["not"]:
            result = not result

        return 0 if result else 1


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(ModelCompareCommand())
