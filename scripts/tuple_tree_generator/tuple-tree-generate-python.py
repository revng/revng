#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys
import yaml
from pathlib import Path

from tuple_tree_generator import Schema
from tuple_tree_generator import generate_python

argparser = argparse.ArgumentParser()
argparser.add_argument("schema", help="YAML schema")
argparser.add_argument("--output", "-o", help="Output to this file (defaults to stdout)")
argparser.add_argument("--namespace", required=True, help="Base namespace for generated types")
argparser.add_argument("--string-type", action="append", help="Treat this type as a string")
argparser.add_argument("--root-type", required=True, help="Schema root type")
argparser.add_argument(
    "--external-type",
    action="append",
    help="Assume these types are externally defined. "
    "They will be imported using `from .external import <name>",
)


def main(args):
    with open(args.schema) as f:
        raw_schema = yaml.safe_load(f)

    schema = Schema(raw_schema, args.namespace)
    source = generate_python(
        schema,
        args.root_type,
        string_types=args.string_type,
        external_types=args.external_type,
    )

    if args.output:
        output_path = Path(args.output)
        # Ensure the containing directory is created
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(source)
    else:
        sys.stdout.write(source)


if __name__ == "__main__":
    parsed_args = argparser.parse_args()
    main(parsed_args)
