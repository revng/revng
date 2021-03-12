#!/usr/bin/env python3

# This script generates C++ classes from a GraphViz file representing a monotone
# framework and a template file.
#
# The graph should meet the same requirements as monotone-framework.py
#

import argparse
import sys
from collections import defaultdict
from pygraphviz import AGraph

import importlib
monotone_framework = importlib.import_module("monotone-framework")
def gen_combine_values(lattice, reachability):
  out = ''
  # Emit the combine operator
  out += """LatticeElement combineValues(const LatticeElement &Lh, const LatticeElement &Rh) {
"""

  node_by_index = lambda index: monotone_framework.get_unique([x
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
      out += ("""  if (""")
    else:
      out += (""" else if (""")
    conditions = []
    for this, other in sorted(pairs, key=lambda x: (x[0].name, x[1].name)):
      condition = "(Lh == LatticeElement::{} && Rh == LatticeElement::{})"
      condition = condition.format(this.name, other.name)
      conditions.append(condition)
    conditions[0] = conditions[0].lstrip()
    conditions_string = ("\n      || "
                         if first
                         else "\n             || ")
    out += conditions_string.join(conditions)
    out += (""") {{
    return LatticeElement::{};
  }}""".format(output.name))
    first = False

  out += ("""
  return Lh;
}

""")
  return out

def gen_is_less_or_equal(lattice, reachability):
  
  out = ''

  # Emit the comparison operator of the lattice
  out += ("""bool isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh) {
  return Lh == Rh
    || """)

  result = []
  for v1 in sorted(lattice.nodes_iter(), key=lambda x: x.name):
    for v2 in sorted(lattice.nodes_iter(), key=lambda x: x.name):
      if v1 != v2:
        i1 = int(v1.attr["index"])
        i2 = int(v2.attr["index"])
        if reachability[i1][i2] != 0:
          condition = """(Lh == LatticeElement::{} && Rh == LatticeElement::{})"""
          condition = condition.format(v1.name, v2.name)
          result.append(condition)

  out += ("\n    || ".join(result))
  out += (""";
}

""")
  return out

def gen_transfer_function(tf_names, transfer_functions):
  # Emit the transfer function implementation
  out = ''
  out += """LatticeElement transfer(TransferFunction T, const LatticeElement &E) {
  switch(T) {
"""
  for tf in tf_names:
    out += ("""  case TransferFunction::{}:
    switch(E) {{
""".format(tf))
    for edge in transfer_functions[tf]:
      source, destination = edge
      out += ("""    case LatticeElement::{}:
      return LatticeElement::{};
""".format(source, destination))
    out += ("""    default:
      return E;
    }
    return E;

""")

  out += ("""  default:
    return E;
  }
}

""")
  return out
def gen_lattice_element_enum(lattice):
  out = ''
  # Print the enumeration of all the possible lattice values
  out += ("""enum LatticeElement {
""")

  # Get all the names
  values = sorted([v.name for v in lattice.nodes_iter()])
  out += ("  " + ",\n  ".join(values) + "\n")
  out += ("""};
""")
  return out
def gen_transfer_function_enum(tf_graph):
  out = ''
  # Print the enumeration of all the possible transfer functions
  out += ("""enum TransferFunction {
""")

  # Get all the transfer function names
  tfs = [e.attr["label"] for e in tf_graph.edges_iter()]
  out += ("  " + ",\n  ".join(sorted(tfs)) + "\n")
  out += ("""};

""")
  return out
def process_graph(path, call_arcs):
  input_graph = AGraph(path)

  monotone_framework.enumerate_graph(input_graph)

  lattice = monotone_framework.extract_lattice(input_graph)

  # Check that the lattice is valid for a monotone framework
  top, bottom = monotone_framework.check_lattice(lattice)

  reachability = monotone_framework.compute_reachability_matrix(lattice)

  tf_graph = monotone_framework.extract_transfer_function_graph(input_graph, call_arcs)

  transfer_functions = defaultdict(lambda: [])
  for edge in tf_graph.edges_iter():
    transfer_functions[edge.attr["label"]].append(edge)

  # Check the monotonicity of the transfer function
  assert monotone_framework.check_transfer_functions(tf_graph, reachability, transfer_functions)
  
  tf_names = sorted([e.attr["label"] for e in tf_graph.edges_iter()])
  
  default_lattice_element = [v.name
            for v in lattice.nodes_iter()
            if v.attr["peripheries"]][0]

  return {
    'transfer_function_names': tf_names,
    'lattice_name': input_graph.name,
    'lattice_elements_enums': gen_lattice_element_enum(lattice),
    'default_lattice_element': default_lattice_element,
    'transfer_function_enums': gen_transfer_function_enum(tf_graph),
    'is_less_or_equal_definition': gen_is_less_or_equal(lattice, reachability),
    'combine_values_definition': gen_combine_values(lattice, reachability),
    'transfer_function_definition': gen_transfer_function(tf_names, transfer_functions)
  }

def main():
  parser = argparse.ArgumentParser(description="Generate C++ code from dot \
  files representing a monotone framework.")
  parser.add_argument("--call-arcs",
                      action="store_true",
                      help="Add call arcs.")
  parser.add_argument("template",
                      metavar="TEMPLATE",
                      help="""Lattice template file.
                      you can use the keywords:
                      - %LatticeName% the name of the lattice extracted from the graph name
                      - %LatticeElement% the C++ enum definition for the elements in the lattice `enum LatticeElement {...}`
                      - %DefaultLatticeElement% the name for the default value for a lattice element 
                      - %TransferFunction% the C++ enum definition for the possible transfer functions `enum TransferFunction {...}`
                      - %isLessOrEqual% the C++ function with signature `bool isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh)`
                      - %combineValues% the C++ function with signature `bool combineValues(const LatticeElement &Lh, const LatticeElement &Rh)`
                      - %transfer% the C++ function with signature `bool transfer(TransferFunction T, const LatticeElement &Rh)`
                      
                      """)
  parser.add_argument("inputs",
                      metavar="GRAPH",
                      nargs="+",
                      help="GraphViz input file.")
  args = parser.parse_args()

  # Print the template file
  with open(args.template) as template_file:
    template = template_file.read()

  # Process each input graph
  all_transfer_functions = set()
  for path in args.inputs:
    generated_code = process_graph(
        path, args.call_arcs)
    all_transfer_functions |= set(generated_code['transfer_function_names'])
  monotone_framework.out(template.replace('%LatticeName%', generated_code['lattice_name'])
                                 .replace('%LatticeElement%', generated_code['lattice_elements_enums'])
                                 .replace('%DefaultLatticeElement%', generated_code['default_lattice_element'])
                                 .replace('%TransferFunction%', generated_code['transfer_function_enums'])
                                 .replace('%isLessOrEqual%', generated_code['is_less_or_equal_definition'])
                                 .replace('%combineValues%', generated_code['combine_values_definition'])
                                 .replace('%transfer%', generated_code['transfer_function_definition']))


  return 0


if __name__ == '__main__':
  sys.exit(main())
