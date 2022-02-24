#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import logging
import shutil
import subprocess
import yaml
from pathlib import Path

from tuple_tree_generator import Schema
from tuple_tree_generator import generate_cpp_headers

argparser = argparse.ArgumentParser()
argparser.add_argument("schema", help="YAML schema")
argparser.add_argument("output_dir", help="Output to this directory")
argparser.add_argument("--namespace", required=True, help="Base namespace for generated types")
argparser.add_argument("--include-path-prefix", required=True, help="Prefixed to include paths")


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


def main(args):
    with open(args.schema) as f:
        raw_schema = yaml.safe_load(f)

    schema = Schema(raw_schema, args.namespace)
    sources = generate_cpp_headers(schema, args.include_path_prefix)

    if is_clang_format_available():
        sources = {name: reformat(source) for name, source in sources.items()}
    else:
        logging.warning("Warning: clang-format is not available, cannot format source")

    output_dir = Path(args.output_dir)
    for output_path, output_source in sources.items():
        full_output_path = output_dir / output_path
        # Ensure the containing directory is created
        full_output_path.parent.mkdir(exist_ok=True)
        full_output_path.write_text(output_source)


if __name__ == "__main__":
    parsed_args = argparser.parse_args()
    main(parsed_args)