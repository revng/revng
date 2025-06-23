#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import abc
import argparse
import logging
import sys
from pathlib import Path
from shutil import which
from subprocess import DEVNULL, check_output, run
from tempfile import NamedTemporaryFile

import yaml

from tuple_tree_generator.generators import CppGenerator, DocsGenerator, JSONSchemaGenerator
from tuple_tree_generator.generators import PythonGenerator, TypeScriptGenerator
from tuple_tree_generator.schema import Schema


class Subcommand(abc.ABC):
    def __init__(self, name: str, subparser, string_types: bool = False, add_output: bool = False):
        self.parser = subparser.add_parser(name)
        self.parser.add_argument("schema", help="YAML schema")
        self.parser.add_argument(
            "--namespace", required=True, help="Base namespace for generated types"
        )
        self.parser.add_argument("--scalar-type", action="append", default=[], help="Scalar type")
        if string_types:
            self.parser.add_argument(
                "--string-type", action="append", default=[], help="Treat this type as a string"
            )
        if add_output:
            self.parser.add_argument(
                "--output", "-o", help="Output to this file (defaults to stdout)"
            )
        self.parser.set_defaults(handler=self.handle)

    @abc.abstractmethod
    def handle(self, args, schema: Schema):
        ...

    @staticmethod
    def handle_single_file_output(output: str | None, source: str):
        if output:
            output_path = Path(output)
            # Ensure the containing directory is created
            output_path.parent.mkdir(exist_ok=True)
            output_path.write_text(source, encoding="utf-8")
        else:
            sys.stdout.write(source)


class CppSubcommand(Subcommand):
    def __init__(self, subparser):
        super().__init__("cpp", subparser)
        self.parser.add_argument("output_dir", help="Output to this directory")
        self.parser.add_argument(
            "--include-path-prefix", required=True, help="Prefixed to include paths"
        )
        self.parser.add_argument(
            "--tracking", action="store_true", help="Emits tracking sequences", default=False
        )

    def handle(self, args, schema: Schema):
        generator = CppGenerator(schema, args.tracking, args.include_path_prefix)
        sources = generator.emit()

        if which("clang-format") is not None:
            sources = {name: self.run_clang_format(source) for name, source in sources.items()}
        else:
            logging.warning("Warning: clang-format is not available, cannot format source")

        output_dir = Path(args.output_dir)
        for output_path, output_source in sources.items():
            full_output_path = output_dir / output_path
            # Ensure the containing directory is created
            full_output_path.parent.mkdir(exist_ok=True)
            full_output_path.write_text(output_source)

    @staticmethod
    def run_clang_format(source: str) -> str:
        """Applies clang-format to the given source"""
        return check_output(["clang-format"], input=source, encoding="utf-8")


class DocsSubcommand(Subcommand):
    def __init__(self, subparser):
        super().__init__("docs", subparser, add_output=True)

    def handle(self, args, schema: Schema):
        generator = DocsGenerator(schema)
        self.handle_single_file_output(args.output, generator.emit_docs())


class JSONSchemaSubcommand(Subcommand):
    def __init__(self, subparser):
        super().__init__("jsonschema", subparser, True, True)
        self.parser.add_argument(
            "--separate-string-type",
            action="append",
            default=[],
            help="Treat this type as a string, but emit a separate definition for it",
        )

    def handle(self, args, schema):
        generator = JSONSchemaGenerator(schema, args.string_type, args.separate_string_type)
        self.handle_single_file_output(args.output, generator.emit_jsonschema())


class PythonSubcommand(Subcommand):
    def __init__(self, subparser):
        super().__init__("python", subparser, True)
        self.parser.add_argument("--output", "-o", required=True, help="Output to this file")
        self.parser.add_argument(
            "--external-type",
            action="append",
            default=[],
            help="Assume these types are externally defined. "
            "They will be imported using `from .external import <name>",
        )
        self.parser.add_argument(
            "--mixins", action="append", default=[], help="Python files containing mixins classes"
        )

    def handle(self, args, schema: Schema):
        generator = PythonGenerator(
            schema, args.output, args.string_type, args.external_type, args.mixins
        )
        self.handle_single_file_output(args.output, generator.emit_python())


class TypescriptSubcommand(Subcommand):
    def __init__(self, subparser):
        super().__init__("typescript", subparser, True, True)
        self.parser.add_argument(
            "--global-name", required=True, help="Name of the top-level object"
        )
        self.parser.add_argument(
            "--external-file", action="append", help="Additional ts file to include"
        )
        self.parser.add_argument(
            "--external-type",
            action="append",
            default=[],
            help="Assume these types are externally defined. ",
        )

    def handle(self, args, schema: Schema):
        generator = TypeScriptGenerator(
            schema,
            args.global_name,
            args.external_file,
            args.string_type,
            args.external_type,
        )
        source = generator.emit_typescript()

        if which("prettier") is not None:
            source = self.prettify(source)

        self.handle_single_file_output(args.output, source)

    @staticmethod
    def prettify(source: str) -> str:
        with NamedTemporaryFile(suffix=".ts") as temp_file:
            source_file = Path(temp_file.name)
            source_file.write_text(source)
            run(["prettier", "--write", temp_file.name], check=True, stdout=DEVNULL)
            new_source = source_file.read_text()
        return new_source


def parse_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(required=True)

    CppSubcommand(subparser)
    DocsSubcommand(subparser)
    JSONSchemaSubcommand(subparser)
    PythonSubcommand(subparser)
    TypescriptSubcommand(subparser)

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.schema, encoding="utf-8") as f:
        raw_schema = yaml.safe_load(f)

    schema = Schema(raw_schema, args.namespace, args.scalar_type)
    args.handler(args, schema)


if __name__ == "__main__":
    main()
