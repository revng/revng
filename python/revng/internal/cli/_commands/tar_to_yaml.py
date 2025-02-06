#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys
from typing import Optional

import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import extract_tar, to_string, to_yaml


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
        parser.add_argument(
            "input",
            type=argparse.FileType("rb+"),
            default=sys.stdin.buffer,
            nargs="?",
            help="Input file (stdin if omitted)",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=argparse.FileType("w"),
            default=sys.stdout,
            nargs="?",
            metavar="FILE",
            help="Output file (stdout if omitted)",
        )

    def run(self, options: Options) -> Optional[int]:
        args = options.parsed_args
        output = yaml.dump(extract_tar(args.input.read(), auto_process))
        args.output.write(output)
        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(TarToYAMLCommand())
