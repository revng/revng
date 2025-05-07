#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
from pathlib import Path

import yaml

from tuple_tree_generator import Schema, generate_python

argparser = argparse.ArgumentParser()
argparser.add_argument("schema", help="YAML schema")
argparser.add_argument("--output", "-o", required=True, help="Output to this file")
argparser.add_argument("--namespace", required=True, help="Base namespace for generated types")
argparser.add_argument(
    "--string-type", action="append", default=[], help="Treat this type as a string"
)
argparser.add_argument("--root-type", required=True, help="Schema root type")
argparser.add_argument(
    "--external-type",
    action="append",
    default=[],
    help="Assume these types are externally defined. "
    "They will be imported using `from .external import <name>",
)
argparser.add_argument("--scalar-type", action="append", default=[], help="Scalar type")
argparser.add_argument(
    "--mixins", action="append", default=[], help="Python files containing mixins classes"
)


def main(args):
    with open(args.schema, encoding="utf-8") as f:
        raw_schema = yaml.safe_load(f)

    schema = Schema(raw_schema, args.root_type, args.namespace, args.scalar_type)
    source = generate_python(
        schema,
        args.root_type,
        output=args.output,
        string_types=args.string_type,
        external_types=args.external_type,
        mixins_paths=args.mixins,
    )

    output_path = Path(args.output)
    # Ensure the containing directory is created
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    parsed_args = argparser.parse_args()
    main(parsed_args)
