#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
from pathlib import Path

import yaml

from tuple_tree_generator import Schema, generate_jsonschema

argparser = argparse.ArgumentParser()
argparser.add_argument("schema", help="YAML schema")
argparser.add_argument("--output", "-o", help="Output to this file")
argparser.add_argument("--namespace", required=True, help="Base namespace for generated types")
argparser.add_argument("--root-type", required=True, help="Schema root type")
argparser.add_argument(
    "--string-type", action="append", default=[], help="Treat this type as a string"
)
argparser.add_argument(
    "--separate-string-type",
    action="append",
    default=[],
    help="Treat this type as a string, but emit a separate definition for it",
)
argparser.add_argument("--scalar-type", action="append", default=[], help="Scalar type")


def main(args):
    with open(args.schema, encoding="utf-8") as f:
        raw_schema = yaml.safe_load(f)

    schema = Schema(raw_schema, args.root_type, args.namespace, args.scalar_type)
    jsonschema_source = generate_jsonschema(
        schema,
        args.root_type,
        string_types=args.string_type,
        separate_string_types=args.separate_string_type,
    )

    if args.output:
        output_file = Path(args.output)
        # Ensure the containing directory is created
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text(jsonschema_source, encoding="utf-8")
    else:
        print(jsonschema_source)


if __name__ == "__main__":
    parsed_args = argparser.parse_args()
    main(parsed_args)
