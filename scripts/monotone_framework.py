#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# This script generates C++ classes from a GraphViz file representing a monotone
# framework.
#
# The graph should have a set of edges with no labels to represent the lattice
# and another set of edges with a label representing the transfer functions.
# Each different label represents a different transfer function.
# Additionally, one of the nodes should have a double border ("peripheries=2")
# to represent it's the initial node.
#
# This script works and should keep working on Python 3.6 .

import argparse

# Standard imports
import sys
from collections import defaultdict
from itertools import product
from typing import List, Mapping, Tuple

# networkx
import networkx

# numpy - optional
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

    def matrix_multiplication(a, b):
        zip_b = list(zip(*b))
        return [[sum(a * b for a, b in zip(row, col)) for col in zip_b] for row in a]


def out(string):
    sys.stdout.write(string)


def log(string):
    sys.stderr.write(string + "\n")


def enumerate_graph(graph: networkx.MultiDiGraph):
    # Assign an identifier to each node
    for i, vertex in enumerate(graph.nodes()):
        graph.nodes[vertex]["index"] = str(i)


def get_unique(container):
    assert len(container) == 1
    return container[0]


def extract_lattice(input_graph: networkx.MultiDiGraph):
    # Compute the lattice graph from the input graph. In practice we remove all
    # the edges that have a label, since they represent a transfer function
    return drop_edge_if(input_graph, lambda edge, edge_data: "label" in edge_data)


def compute_reachability_matrix(graph: networkx.MultiDiGraph):
    # Build the adjacency matrix
    vertices = graph.number_of_nodes()
    edges = graph.number_of_edges()
    adj = [[float(0) for x in range(vertices)] for y in range(vertices)]

    # Self-loop
    for x in range(vertices):
        adj[x][x] = 1

    # Set to 1 all the cells representing a pair of adjacent vertices
    for src, dst in graph.edges():
        x = int(graph.nodes[src]["index"])
        y = int(graph.nodes[dst]["index"])
        adj[x][y] = 1

    # Build the reachability matrix
    # TODO: reimplement re-enumerating all the elements in the matrix instead of
    #       using float
    if np is not None:
        reachability = (np.matrix(adj) ** edges).tolist()
    else:
        reachability = adj
        for _ in range(edges):
            reachability = matrix_multiplication(reachability, adj)

    for x in range(vertices):
        reachability[x][x] = float("Inf")

    return reachability


def is_cyclic(graph: networkx.MultiDiGraph, entry):
    visited = set()
    path = [object()]
    path_set = set(path)
    stack = [graph.successors(entry)]
    while stack:
        for v in stack[-1]:
            if v in path_set:
                return True
            if v not in visited:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                stack.append(iter(graph.successors(v)))
                break
        else:
            path_set.remove(path.pop())
            stack.pop()
    return False


def check_lattice(lattice: networkx.MultiDiGraph):
    # Check we have a single top and a single bottom
    top = None
    bottom = None
    for v in lattice.nodes():
        if lattice.in_degree(v) == 0:
            assert bottom is None
            bottom = v

        if lattice.out_degree(v) == 0:
            assert top is None
            top = v
    assert top is not None
    assert bottom is not None

    # Check for the absence of back edges
    assert not is_cyclic(lattice, bottom)

    return top, bottom


def drop_edge_if(graph: networkx.MultiDiGraph, condition):
    new_graph = graph.copy()

    for src, dst, key, edge_data in list(new_graph.edges(keys=True, data=True)):
        if condition((src, dst), edge_data):
            new_graph.remove_edge(src, dst, key)
    return new_graph


def extract_transfer_function_graph(input_graph: networkx.MultiDiGraph, call_arcs):
    # Build the transfer function graph. Drop all the edges without a label (those
    # representing the lattice).
    tf_graph = drop_edge_if(input_graph, lambda edge, edge_data: "label" not in edge_data)

    # Add the ReturnFrom... arcs
    if call_arcs:
        edge_labels = {edge_data["label"] for _, _, edge_data in tf_graph.edges(data=True)}
        for v in tf_graph.nodes():

            return_edge_name = "ReturnFrom" + v

            # Check if the ReturnFrom edge has already been explicitly provided
            if return_edge_name in edge_labels:
                continue

            # All the incoming edges must come from the same source
            source = {edge[0] for edge in tf_graph.in_edges(v)}
            assert len(source) <= 1
            if source:
                source = list(source)[0]
                edge_key = tf_graph.add_edge(source, v)
                new_edge = tf_graph.edges[(source, v, edge_key)]
            else:
                edge_key = tf_graph.add_edge(v, v)
                new_edge = tf_graph.edges[(v, v, edge_key)]
            new_edge["label"] = return_edge_name

    return tf_graph


def check_transfer_functions(
    tf_graph: networkx.MultiDiGraph,
    reachability: List[List[float]],
    transfer_functions: Mapping[str, List[Tuple[str, str]]],
):
    tf_graph = tf_graph.copy()
    # Automatically add self-loops where required
    all_vertices = set(tf_graph.nodes())
    for name, edges in transfer_functions.items():
        no_self = set()
        for source, _ in edges:
            no_self.add(source)
        need_self = all_vertices - no_self
        for vertex in need_self:
            tf_graph.add_edge(vertex, vertex, label=name)
            edges.append((vertex, vertex))

    result = True

    def s(edge):
        return int(tf_graph.nodes[edge[0]]["index"])

    def d(edge):
        return int(tf_graph.nodes[edge[1]]["index"])

    def lte(a, b):
        return reachability[a][b] != 0

    for name, edges in transfer_functions.items():
        for a, b in product(edges, edges):
            if a == b:
                continue
            if lte(s(a), s(b)) and not lte(d(a), d(b)):
                result = False
                log(f"The transfer function {name} is not monotone:\n")
                log(f"  {a[0].name} -> {a[1].name}\n  {b[0].name} -> {b[1].name}\n")  # type: ignore

    return result


def load_graph(path: str) -> networkx.MultiDiGraph:
    """
    Loads a networkx.MultiDiGraph from a dot file
    Notes:
    - pydot treats every attribute as a string, so we need to manually unquote
      strings
    """
    graph = networkx.MultiDiGraph(networkx.nx_pydot.read_dot(path))
    for _, _, data in graph.edges(data=True):
        for key, val in data.items():
            if isinstance(val, str) and len(val) > 2 and val[0] == '"' and val[-1] == '"':
                val = val[1:-1]
                data[key] = val
    return graph


def process_graph(path, call_arcs):
    out_str = ""
    input_graph = load_graph(path)

    enumerate_graph(input_graph)

    lattice = extract_lattice(input_graph)

    # Check that the lattice is valid for a monotone framework
    top, bottom = check_lattice(lattice)

    reachability = compute_reachability_matrix(lattice)

    tf_graph = extract_transfer_function_graph(input_graph, call_arcs)

    transfer_functions = defaultdict(lambda: [])
    for src, dst, edge_data in tf_graph.edges(data=True):
        transfer_functions[edge_data["label"]].append((src, dst))

    # Check the monotonicity of the transfer function
    assert check_transfer_functions(tf_graph, reachability, transfer_functions)

    name = input_graph.name
    # Generate C++ class
    out_str += f"""class {name} {{
public:
"""

    # Print the enumeration of all the possible lattice values
    out_str += """  enum Values {
"""

    # Get the default lattice element (has the "peripheries" property)
    default = [v for v, v_data in lattice.nodes(data=True) if "peripheries" in v_data][0]
    # Get all the names
    values = sorted(v for v in lattice.nodes())
    out_str += "    " + ",\n    ".join(values) + "\n"
    out_str += """  };

"""
    # Print the enumeration of all the possible transfer functions
    out_str += """  enum TransferFunction {{
"""

    # Get all the transfer function names
    tfs = [e_data["label"] for _, _, e_data in tf_graph.edges(data=True)]
    out_str += "    " + ",\n    ".join(sorted(tfs)) + "\n"
    out_str += """  };

"""

    # Emit constructors and factory method for default value (i.e., the initial
    # one)
    out_str += f"""public:
  {name}() :
    Value({bottom}) {{ }}

  {name}(Values V) :
    Value(V) {{ }}

  static Values initial() {{
    return {default};
  }}

"""

    # Emit the combine operator
    out_str += f"""  void combine(const {name} &Other) {{
"""

    def node_by_index(index):
        return get_unique(
            [x for x, x_data in lattice.nodes(data=True) if x_data["index"] == str(index)]
        )

    result = defaultdict(lambda: [])
    for v1, v1_data in lattice.nodes(data=True):
        for v2, v2_data in lattice.nodes(data=True):
            if v1 != v2:
                i1 = int(v1_data["index"])
                i2 = int(v2_data["index"])

                def nonzero(i):
                    return {x[0] for x in enumerate(reachability[i]) if x[1] != 0}

                output = max(
                    nonzero(i1) & nonzero(i2),
                    key=lambda i: reachability[i1][i] + reachability[i2][i],
                )
                output = node_by_index(output)

                result[output].append((v1, v2))
                assert output in result

    first = True
    for output, pairs in sorted(result.items(), key=lambda x: x[0]):
        if first:
            out_str += """    if ("""
        else:
            out_str += """ else if ("""
        conditions = []
        for this, other in sorted(pairs, key=lambda x: (x[0], x[1])):
            condition = "(Value == {} && Other.Value == {})"
            condition = condition.format(this, other)
            conditions.append(condition)
        conditions[0] = conditions[0].lstrip()
        conditions_string = "\n        || " if first else "\n               || "
        out_str += conditions_string.join(conditions)
        out_str += f""") {{
      Value = {output};
    }}"""
        first = False

    out_str += """
  }

"""

    # Emit the comparison operator of the lattice
    out_str += f"""  bool lowerThanOrEqual(const {name} &Other) const {{
    return Value == Other.Value
      || """

    result = []
    for v1 in sorted(lattice.nodes()):
        for v2 in sorted(lattice.nodes()):
            if v1 != v2:
                i1 = int(lattice.nodes[v1]["index"])
                i2 = int(lattice.nodes[v2]["index"])
                if reachability[i1][i2] != 0:
                    condition = f"""(Value == {v1} && Other.Value == {v2})"""
                    result.append(condition)

    out_str += "\n      || ".join(result)
    out_str += """;
  }

"""

    tf_names = sorted([e_data["label"] for _, _, e_data in tf_graph.edges(data=True)])

    # Emit the transfer function implementation
    out_str += """  void transfer(TransferFunction T) {{
    switch(T) {{
"""
    for tf in tf_names:
        out_str += f"""    case {tf}:
      switch(Value) {{
"""
        for edge in transfer_functions[tf]:
            source, destination = edge
            out_str += f"""      case {source}:
        Value = {destination};
        break;
"""
        out_str += """      default:
        break;
      }
      break;

"""

    out_str += """    }
  }

"""

    # Emit the transfer function implementation
    out_str += """  void transfer(GeneralTransferFunction T) {{
    switch(T) {{
"""
    for tf in tf_names:
        out_str += f"""    case GeneralTransferFunction::{tf}:
      switch(Value) {{
"""
        for edge in transfer_functions[tf]:
            source, destination = edge
            out_str += f"""      case {source}:
        Value = {destination};
        break;
"""
        out_str += """      default:
        break;
      }
      break;

"""

    out_str += """    default:
      revng_abort();
    }

  }

"""

    if call_arcs:
        # Emit the method implementing the transfer function for returning from a
        # call
        out_str += """  TransferFunction returnTransferFunction() const {{
    switch(Value) {{
"""

        for value in values:
            out_str += f"""    case {value}:
      return ReturnFrom{value};
"""
        out_str += """    }

    revng_abort();
  }

"""

    # Emit a method to return the name of analysis
    out_str += f"""  static const char *name() {{
    return "{name}";
  }}

"""

    # Emit a method returning the top element
    out_str += f"""  static {name} top() {{
    return {name}({top});
  }}

"""

    # Accessor for the current value
    out_str += """  Values value() const { return Value; }

"""

    # Debug methods
    out_str += """  void dump() const {{ dump(dbg); }}

  template<typename T>
  void dump(T &Output) const {{
    switch(Value) {{
"""

    for value in values:
        out_str += f"""    case {value}:
      Output << "{value}";
      break;
"""
    out_str += """    }
  }

"""

    # Emit private data
    out_str += """private:
  Values Value;
}};

"""

    return tf_names, out_str


def main():
    parser = argparse.ArgumentParser(
        description="Generate C++ code from dot \
    files representing a monotone framework."
    )
    parser.add_argument("--call-arcs", action="store_true", help="Add call arcs.")
    parser.add_argument("header", metavar="HEADER", help="C++ file header.")
    parser.add_argument("inputs", metavar="GRAPH", nargs="+", help="GraphViz input file.")
    args = parser.parse_args()

    # Print the header file
    with open(args.header, encoding="utf-8") as header_file:
        out(header_file.read())

    # Process each input graph
    result = ""
    all_transfer_functions = set()
    for path in args.inputs:
        transfer_functions, output = process_graph(path, args.call_arcs)
        result += output
        all_transfer_functions |= set(transfer_functions)

    out(
        """enum class GeneralTransferFunction {
  """
    )
    out(
        """,
  """.join(
            sorted(all_transfer_functions)
        )
    )
    out(
        """,
  InvalidTransferFunction
};

"""
    )

    out(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
