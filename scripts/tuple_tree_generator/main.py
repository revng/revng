#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import shutil
import subprocess
import yaml
from pathlib import Path

from tuple_tree_generator.generators import CppHeadersGenerator, JSONSchemaGenerator

argparser = argparse.ArgumentParser()
argparser.add_argument("schema", help="YAML schema")
argparser.add_argument("output_dir", help="Output to this directory")
argparser.add_argument("--namespace", required=True, help="Base namespace for generated types")
output_format_group = argparser.add_mutually_exclusive_group(required=True)
output_format_group.add_argument("--cpp-headers", action="store_true", help="Emit C++ headers")
argparser.add_argument("--include-path-prefix", help="Prefixed to include paths. Only allowed with --cpp-headers")
output_format_group.add_argument("--jsonschema", action="store_true", help="Emit a jsonschema")
argparser.add_argument("--jsonschema-root-type", help="JSONSchema root type (mandatory with --jsonschema)")


def is_clang_format_available():
    return shutil.which("clang-format")


def reformat(source: str) -> str:
    """Applies clang-format to the given source"""
    return subprocess.check_output(
        [
            "clang-format",
        ],
        input=source,
        encoding="utf-8",
    )


def validate_args(args):
    if args.jsonschema and not args.jsonschema_root_type:
        argparser.error("--jsonschema-root-type is mandatory with --jsonschema")


def main(args):
    validate_args(args)

    with open(args.schema) as f:
        schema = yaml.safe_load(f)

    if args.cpp_headers:
        if args.include_path_prefix is None:
            raise argparse.ArgumentError("You must provide --include-path-prefix when using --cpp-headers")

        generator = CppHeadersGenerator(schema, args.namespace, user_include_path=args.include_path_prefix)
        sources = generator.emit()

        if is_clang_format_available():
            sources = {name: reformat(source) for name, source in sources.items()}
        else:
            print("Warning: clang-format is not available, cannot format source")
    elif args.jsonschema:
        generator = JSONSchemaGenerator(schema, args.namespace, args.jsonschema_root_type)
        sources = generator.emit()
    else:
        argparser.error("Provide --cpp-headers or --jsonschema")

    output_dir = Path(args.output_dir)
    for output_path, output_source in sources.items():
        full_output_path = output_dir / output_path
        # Ensure the containing directory is created
        full_output_path.parent.mkdir(exist_ok=True)
        full_output_path.write_text(output_source)


if __name__ == "__main__":
    parsed_args = argparser.parse_args()
    main(parsed_args)
