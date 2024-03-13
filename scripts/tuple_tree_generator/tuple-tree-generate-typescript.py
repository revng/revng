#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
from pathlib import Path
from subprocess import DEVNULL, run
from tempfile import NamedTemporaryFile

import yaml

from tuple_tree_generator import Schema, generate_typescript

argparser = argparse.ArgumentParser()
argparser.add_argument("schema", help="YAML schema")
argparser.add_argument("--output", "-o", help="Output to this file")
argparser.add_argument("--namespace", required=True, help="Base namespace for generated types")
argparser.add_argument("--root-type", required=True, help="Schema root type")
argparser.add_argument("--global-name", required=True, help="Name of the top-level object")
argparser.add_argument("--prettier", help="Path to the prettier binary for formatting")
argparser.add_argument("--external-file", action="append", help="Additional ts file to include")
argparser.add_argument(
    "--string-type", action="append", default=[], help="Treat this type as a string"
)
argparser.add_argument(
    "--external-type",
    action="append",
    default=[],
    help="Assume these types are externally defined. ",
)
argparser.add_argument("--scalar-type", action="append", default=[], help="Scalar type")


def main(args):
    with open(args.schema) as f:
        raw_schema = yaml.safe_load(f)

    schema = Schema(raw_schema, args.namespace, args.scalar_type)
    source = generate_typescript(
        schema,
        args.root_type,
        args.global_name,
        args.external_file,
        args.string_type,
        args.external_type,
    )

    if args.prettier is not None:
        source = prettify_typescript(source, args.prettier)

    if args.output:
        output_file = Path(args.output)
        # Ensure the containing directory is created
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text(source)
    else:
        print(source)


def prettify_typescript(source: str, prettier: str) -> str:
    with NamedTemporaryFile(suffix=".ts") as temp_file:
        source_file = Path(temp_file.name)
        source_file.write_text(source)
        run([prettier, "--write", temp_file.name], check=True, stdout=DEVNULL)
        new_source = source_file.read_text()
    return new_source


if __name__ == "__main__":
    parsed_args = argparser.parse_args()
    main(parsed_args)
