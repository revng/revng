#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
from typing import Optional

import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import extract_tar, file_wrapper, to_string, to_yaml


def auto_process(filename: str, raw: bytes) -> str:
    # TODO: add base64 for non-text stuff
    if filename.endswith(".yml"):
        return to_yaml(filename, raw)
    else:
        return to_string(filename, raw)


class TarToYAMLCommand(Command):
    def __init__(self):
        super().__init__(("tar", "to-yaml"), "Turn a tar archive into YAML")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.description = "Turn a tar archive into YAML"
        parser.add_argument("input", nargs="?", help="Input file (stdin if omitted)")
        parser.add_argument(
            "-o", "--output", nargs="?", metavar="FILE", help="Output file (stdout if omitted)"
        )

    def run(self, options: Options) -> Optional[int]:
        args = options.parsed_args
        with file_wrapper(args.input, "rb") as input_file:
            output = yaml.dump(extract_tar(input_file.read(), auto_process))
        with file_wrapper(args.output, "w") as output_file:
            output_file.write(output)
        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(TarToYAMLCommand())
