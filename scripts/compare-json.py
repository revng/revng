#!/usr/bin/env python3

# This script compares two JSON files. In general, the reference JSON file is
# supposed to be included in the other one. For instance, {"a": 1, "b": 2}
# contains {"b": 2}. This is useful for enforcing the content of certain parts
# of a JSON file, in particular during testing.
#
# This script works and should keep working on Python 2.7 and Python 3.

import argparse
import json
import sys

def do_log(indent, message, *args):
  message = message.replace("\n", "\n" + "  " * indent)
  sys.stderr.write(("  " * indent) + message.format(*args) + "\n")

def to_json(obj):
  return json.dumps(obj)

def compare(reference, input, order, exact, verbose, quiet, indent):
  def log(message, *args):
    if not quiet or verbose:
      do_log(indent, message, *args)

  def vlog(message, *args):
    if verbose:
      log(message, *args)

  recurse = lambda new_reference, new_input, force_quiet: compare(new_reference,
                                                                  new_input,
                                                                  order,
                                                                  exact,
                                                                  verbose,
                                                                  force_quiet,
                                                                  indent + 1)

  vlog("Comparing:\n  {}\n  {}", to_json(reference), to_json(input))

  if type(reference) is not type(input):
    log("Different types met:\n  {}\n  {}",
        to_json(type(reference)),
        to_json(type(input)))
    return False
  elif not exact and type(reference) is dict:
    for key, value in reference.items():
      if key not in input:
        log("Couldn't find the following key in a dictionary: {}",
            to_json(key))
        return False
      else:
        if not recurse(value, input[key], quiet):
          return False
    vlog("Found!")
    return True
  elif not exact and type(reference) is list:
    if order:
      start = 0
      for reference_index, reference_item in enumerate(reference):
        found = False
        for input_index, input_item in enumerate(input[start:]):
          if recurse(reference_item, input_item, True):
            found = True
            start = start + input_index + 1
            break
        if not found:
          log("Couldn't find element {} of a list in order:\n  {}",
              reference_index,
              to_json(reference_item))
          return False
      vlog("Found!")
      return True
    else:
      for reference_index, reference_item in enumerate(reference):
        found = False
        for input_item in input:
          if recurse(reference_item, input_item, True):
            found = True
            break
        if not found:
          log("Couldn't find element {} of a list:\n  {}",
              reference_index,
              to_json(reference_item))
          return False
      vlog("Found!")
      return True
  else:
    if reference != input:
      log("Difference:\n  {}\n  {}", to_json(reference), to_json(input))
      return False
    vlog("Found!")
    return True

def main():
  parser = argparse.ArgumentParser(description='Compare a JSON file against a \
    reference.')
  parser.add_argument('reference',
                      metavar='REFERENCE',
                      help='The reference file.')
  parser.add_argument('input', metavar='INPUT',
                      default="/dev/stdin",
                      help='The input file.')
  parser.add_argument('--exact',
                      action='store_true',
                      help=("Match exactly, containing the reference is not "
                            + "enough."))
  parser.add_argument('--order',
                      action='store_true',
                      help="The order of elements of a list matters.")
  parser.add_argument('--verbose',
                      action='store_true',
                      help="Print each element being compared.")
  args = parser.parse_args()

  with open(args.reference) as reference_file, open(args.input) as input_file:
    reference = json.load(reference_file)
    input = json.load(input_file)

  result = compare(reference,
                   input,
                   args.order,
                   args.exact,
                   args.verbose,
                   False,
                   0)

  return 0 if result else 1

if __name__ == "__main__":
  sys.exit(main())
