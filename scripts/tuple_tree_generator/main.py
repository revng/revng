#!/usr/bin/env python3
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import shutil
import subprocess
import yaml
from pathlib import Path

from tuple_tree_generator.generators import CppHeadersGenerator

argparser = argparse.ArgumentParser()
argparser.add_argument("schema", help="YAML schema")
argparser.add_argument("output_dir", help="Output to this directory")
argparser.add_argument("--namespace", required=True, help="Base namespace for generated types")
output_format_group = argparser.add_mutually_exclusive_group(required=True)
output_format_group.add_argument("--cpp-headers", action="store_true", help="Emit C++ headers")


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
        schema = yaml.safe_load(f)

    if args.cpp_headers:
        generator = CppHeadersGenerator(schema, args.namespace)
        sources = generator.emit()

        if is_clang_format_available():
            sources = {name: reformat(source) for name, source in sources.items()}
        else:
            print("Warning: clang-format is not available, cannot format source")
    else:
        argparser.error("Provide --cpp-headers")

    output_dir = Path(args.output_dir)
    for output_path, output_source in sources.items():
        full_output_path = output_dir / output_path
        # Ensure the containing directory is created
        full_output_path.parent.mkdir(exist_ok=True)
        full_output_path.write_text(output_source)


if __name__ == "__main__":
    parsed_args = argparser.parse_args()
    main(parsed_args)
