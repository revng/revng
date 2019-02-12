#!/usr/bin/env python3

# This script generates C++ classes from a GraphViz file representing a monotone
# framework.
#
# The graph should have a set of edges with no labels to represent the lattice
# and another set of edges with a label representing the transfer functions.
# Each different label represents a different transfer function.
# Additionally, one of the nodes should have a double border ("peripheries=2")
# to represent it's the initial node.
#
# This script works and should keep working on Python 2.7 and Python 3.

# Standard imports
import sys
import argparse
import re
import tempfile
from itertools import product
from collections import defaultdict

# pygraphviz
from pygraphviz import AGraph

# numpy (optional)
try:
  import numpy as np
except ImportError:
  np = None
  def matrix_multiplication(a, b):
    zip_b = list(zip(*b))
    return [[sum(a * b for a, b in zip(row, col))
             for col in zip_b]
            for row in a]

out = lambda string: sys.stdout.write(string)
log = lambda string: sys.stderr.write(string + "\n")

def enumerate_graph(graph):
  # Assign an identifier to each node
  index = 0
  for vertex in graph.nodes_iter():
    vertex.attr["index"] = str(index)
    index += 1

def get_unique(container):
  assert len(container) == 1
  return container[0]

def extract_lattice(input_graph):
  # Compute the lattice graph from the input graph. In practice we remove all
  # the edges that have a label, since they represent a transfer function
  return drop_edge_if(input_graph, lambda edge: edge.attr["label"])

def compute_reachability_matrix(graph):
  # Build the adjacency matrix
  vertices = graph.number_of_nodes()
  edges = graph.number_of_edges()
  adj = [[float(0) for x in range(vertices)] for y in range(vertices)]

  # Self-loop
  for x in range(vertices):
    adj[x][x] = 1

  # Set to 1 all the cells representing a pair of adjacent vertices
  for edge in graph.edges_iter():
    x = int(edge[0].attr["index"])
    y = int(edge[1].attr["index"])
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

def is_cyclic(graph, entry):
  visited = set()
  path = [object()]
  path_set = set(path)
  stack = [graph.successors_iter(entry)]
  while stack:
    for v in stack[-1]:
      if v in path_set:
        return True
      elif v not in visited:
        visited.add(v)
        path.append(v)
        path_set.add(v)
        stack.append(iter(graph.successors_iter(v)))
        break
    else:
      path_set.remove(path.pop())
      stack.pop()
  return False

def check_lattice(lattice):
  # Check we have a single top and a single bottom
  top = None
  bottom = None
  for v in lattice.nodes_iter():
    if lattice.in_degree(v) == 0:
      assert bottom == None
      bottom = v

    if lattice.out_degree(v) == 0:
      assert top == None
      top = v
  assert top is not None
  assert bottom is not None

  # Check for the absence of back edges
  assert not is_cyclic(lattice, bottom)

  return top, bottom

def drop_edge_if(graph, condition):
  new_graph = graph.copy()

  for edge in new_graph.edges():
    new_graph.remove_edge(edge)

  for edge in graph.edges_iter():
    if not condition(edge):
      new_graph.add_edge(edge[0], edge[1], None, **edge.attr)

  return new_graph

def extract_transfer_function_graph(input_graph, call_arcs):
  # Build the transfer function graph. Drop all the edges without a label (those
  # representing the lattice).
  tf_graph = drop_edge_if(input_graph, lambda edge: not edge.attr["label"])

  # Add the ReturnFrom... arcs
  if call_arcs:
    edge_labels = set(edge.attr["label"] for edge in tf_graph.edges())
    for v in tf_graph.nodes_iter():

      return_edge_name = "ReturnFrom" + v.name

      # Check if the ReturnFrom edge has already been explicitly provided
      if return_edge_name in edge_labels:
        continue

      # All the incoming edges must come from the same source
      source = set(edge[0] for edge in tf_graph.in_edges(v))
      assert len(source) <= 1
      if source:
        source = list(source)[0]
        tf_graph.add_edge(source, v)
        new_edge = tf_graph.get_edge(source, v)
      else:
        tf_graph.add_edge(v, v)
        new_edge = tf_graph.get_edge(v, v)
      new_edge.attr["label"] = return_edge_name

  return tf_graph

def check_transfer_functions(tf_graph, reachability, transfer_functions):
  tf_graph = tf_graph.copy()
  # Automatically add self-loops where required
  all_vertices = set(tf_graph.nodes())
  for name, edges in transfer_functions.items():
    no_self = set()
    for edge in edges:
      no_self.add(edge[0])
    need_self = all_vertices - no_self
    for vertex in need_self:
      tf_graph.add_edge(vertex, vertex)
      new_edge = tf_graph.get_edge(vertex, vertex)
      new_edge.attr["label"] = name
      edges.append(new_edge)

  result = True
  s = lambda edge: int(edge[0].attr["index"])
  d = lambda edge: int(edge[1].attr["index"])
  lte = lambda a, b: reachability[a][b] != 0
  for name, edges in transfer_functions.items():
    for a, b in product(edges, edges):
      if a == b:
        continue

      if lte(s(a), s(b)) and not lte(d(a), d(b)):
        result = False
        log("The transfer function {} is not monotone:\n".format(name))
        log("  {} -> {}\n  {} -> {}\n".format(a[0].name,
                                              a[1].name,
                                              b[0].name,
                                              b[1].name))

  return result

def process_graph(path, call_arcs):
  out = ""
  input_graph = AGraph(path)

  enumerate_graph(input_graph)

  lattice = extract_lattice(input_graph)

  # Check that the lattice is valid for a monotone framework
  top, bottom = check_lattice(lattice)

  reachability = compute_reachability_matrix(lattice)

  tf_graph = extract_transfer_function_graph(input_graph, call_arcs)

  transfer_functions = defaultdict(lambda: [])
  for edge in tf_graph.edges_iter():
    transfer_functions[edge.attr["label"]].append(edge)

  # Check the monotonicity of the transfer function
  assert check_transfer_functions(tf_graph, reachability, transfer_functions)

  name = input_graph.name

  # Generate C++ class
  out += ("""class {} {{
public:
""".format(name))

  # Print the enumeration of all the possible lattice values
  out += ("""  enum Values {
""")

  # Get the default lattice element (has the "peripheries" property)
  default = [v.name
             for v in lattice.nodes_iter()
             if v.attr["peripheries"]][0]

  # Get all the names
  values = sorted([v.name for v in lattice.nodes_iter()])
  out += ("    " + ",\n    ".join(values) + "\n")
  out += ("""  };

""")

  # Print the enumeration of all the possible transfer functions
  out += ("""  enum TransferFunction {{
""".format())

  # Get all the transfer function names
  tfs = [e.attr["label"] for e in tf_graph.edges_iter()]
  out += ("    " + ",\n    ".join(sorted(tfs)) + "\n")
  out += ("""  };

""")

  # Emit constructors and factory method for default value (i.e., the initial
  # one)
  out += ("""public:
  {}() :
    Value({}) {{ }}

  {}(Values V) :
    Value(V) {{ }}

  static Values initial() {{
    return {};
  }}

""".format(name, bottom.name, name, default))

  # Emit the combine operator
  out += ("""  void combine(const {} &Other) {{
""".format(name))

  node_by_index = lambda index: get_unique([x
                                            for x in lattice.nodes_iter()
                                            if x.attr["index"] == str(index)])

  result = defaultdict(lambda: [])
  for v1 in lattice.nodes_iter():
    for v2 in lattice.nodes_iter():
      if v1 != v2:
        i1 = int(v1.attr["index"])
        i2 = int(v2.attr["index"])
        nonzero = lambda i: set([x[0]
                                 for x in enumerate(reachability[i])
                                 if x[1] != 0])

        output = max(nonzero(i1) & nonzero(i2),
                     key=lambda i: reachability[i1][i] + reachability[i2][i])
        output = node_by_index(output)

        result[output].append((v1, v2))
        assert output in result

  first = True
  for output, pairs in sorted(result.items(), key=lambda x: x[0]):
    if first:
      out += ("""    if (""")
    else:
      out += (""" else if (""")
    conditions = []
    for this, other in sorted(pairs, key=lambda x: (x[0].name, x[1].name)):
      condition = "(Value == {} && Other.Value == {})"
      condition = condition.format(this.name, other.name)
      conditions.append(condition)
    conditions[0] = conditions[0].lstrip()
    conditions_string = ("\n        || "
                         if first
                         else "\n               || ")
    out += conditions_string.join(conditions)
    out += (""") {{
      Value = {};
    }}""".format(output.name))
    first = False

  out += ("""
  }

""")

  # Emit the comparison operator of the lattice
  out += ("""  bool greaterThan(const {} &Other) const {{
    return !lowerThanOrEqual(Other);
  }}

  bool lowerThanOrEqual(const {} &Other) const {{
    return Value == Other.Value
      || """.format(name, name))

  result = []
  for v1 in sorted(lattice.nodes_iter(), key=lambda x: x.name):
    for v2 in sorted(lattice.nodes_iter(), key=lambda x: x.name):
      if v1 != v2:
        i1 = int(v1.attr["index"])
        i2 = int(v2.attr["index"])
        if reachability[i1][i2] != 0:
          condition = """(Value == {} && Other.Value == {})"""
          condition = condition.format(v1.name, v2.name)
          result.append(condition)

  out += ("\n      || ".join(result))
  out += (""";
  }

""")

  tf_names = sorted([e.attr["label"] for e in tf_graph.edges_iter()])

  # Emit the transfer function implementation
  out += ("""  void transfer(TransferFunction T) {{
    switch(T) {{
""".format(name))
  for tf in tf_names:
    out += ("""    case {}:
      switch(Value) {{
""".format(tf))
    for edge in transfer_functions[tf]:
      out += ("""      case {}:
        Value = {};
        break;
""".format(edge[0].name, edge[1].name))
    out += ("""      default:
        break;
      }
      break;

""")

  out += ("""    }
  }

""")

  # Emit the transfer function implementation
  out += ("""  void transfer(GeneralTransferFunction T) {{
    switch(T) {{
""".format(name))
  for tf in tf_names:
    out += ("""    case GeneralTransferFunction::{}:
      switch(Value) {{
""".format(tf))
    for edge in transfer_functions[tf]:
      out += ("""      case {}:
        Value = {};
        break;
""".format(edge[0].name, edge[1].name))
    out += ("""      default:
        break;
      }
      break;

""")

  out += ("""    default:
      revng_abort();
    }

  }

""")

  if call_arcs:
    # Emit the method implementing the transfer function for returning from a
    # call
    out += ("""  TransferFunction returnTransferFunction() const {{
    switch(Value) {{
""".format())

    for value in values:
      out += ("""    case {}:
      return ReturnFrom{};
""".format(value, value))
    out += ("""    }

    revng_abort();
  }

""")

  # Emit a method to return the name of analysis
  out += ("""  static const char *name() {{
    return "{}";
  }}

""".format(name))

  # Emit a method returning the top element
  out += ("""  static {} top() {{
    return {}({});
  }}

""".format(name, name, top.name))

  # Accessor for the current value
  out += ("""  Values value() const { return Value; }

""")

  # Debug methods
  out += ("""  void dump() const {{ dump(dbg); }}

  template<typename T>
  void dump(T &Output) const {{
    switch(Value) {{
""".format(name, name, name, top.name))

  for value in values:
    out += ("""    case {}:
      Output << "{}";
      break;
""".format(value, value))
  out += ("""    }
  }

""")

  # Emit private data
  out += ("""private:
  Values Value;
}};

""".format())

  return tf_names, out

def main():
  parser = argparse.ArgumentParser(description="Generate C++ code from dot \
    files representing a monotone framework.")
  parser.add_argument("--call-arcs",
                      action="store_true",
                      help="Add call arcs.")
  parser.add_argument("header",
                      metavar="HEADER",
                      help="C++ file header.")
  parser.add_argument("footer",
                      metavar="FOOTER",
                      help="C++ file footer.")
  parser.add_argument("inputs",
                      metavar="GRAPH",
                      nargs="+",
                      help="GraphViz input file.")
  args = parser.parse_args()

  # Print the header file
  with open(args.header) as header_file:
    out(header_file.read())

  # Process each input graph
  result = ""
  all_transfer_functions = set()
  for path in args.inputs:
    transfer_functions, output = process_graph(path, args.call_arcs)
    result += output
    all_transfer_functions |= set(transfer_functions)

  out("""enum class GeneralTransferFunction {
  """)
  out(""",
  """.join(sorted(all_transfer_functions)))
  out(""",
  InvalidTransferFunction
};

""")

  out(result)

  # Print the footer file
  with open(args.footer) as footer_file:
    out(footer_file.read())

  return 0

if __name__ == "__main__":
  sys.exit(main())
