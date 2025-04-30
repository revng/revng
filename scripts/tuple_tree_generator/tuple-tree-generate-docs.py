#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys
from pathlib import Path

import yaml

from tuple_tree_generator import Schema, generate_docs

argparser = argparse.ArgumentParser()
argparser.add_argument("schema", help="YAML schema")
argparser.add_argument("--namespace", required=True, help="Base namespace for generated types")
argparser.add_argument("--output", "-o", help="Output to this file (defaults to stdout)")
argparser.add_argument("--root-type", required=True, help="Schema root type")
argparser.add_argument("--scalar-type", action="append", default=[], help="Scalar type")


def main(args):
    with open(args.schema, encoding="utf-8") as f:
        raw_schema = yaml.safe_load(f)

    schema = Schema(raw_schema, args.root_type, args.namespace, args.scalar_type)
    source = generate_docs(schema, args.root_type)

    if args.output:
        output_path = Path(args.output)
        # Ensure the containing directory is created
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(source, encoding="utf-8")
    else:
        sys.stdout.write(source)


if __name__ == "__main__":
    parsed_args = argparser.parse_args()
    main(parsed_args)
