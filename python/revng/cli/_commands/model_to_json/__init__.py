#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys

import yaml

from revng.cli.commands_registry import Command, CommandsRegistry, Options

from .remap import remap_metaaddress


class ModelToJsonCommand(Command):
    def __init__(self):
        super().__init__(("model", "to-json"), "Extract and process rev.ng model")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--remap",
            action="store_true",
            help="Remap MetaAddresses",
        )

    def run(self, options: Options) -> int:
        args = options.parsed_args

        # Consume YAML generated from revng-efa-extractcfg
        input_file = sys.stdin

        # Decode YAML
        parsed_text = yaml.load(input_file, Loader=yaml.SafeLoader)

        # Remap MetaAddress
        if args.remap:
            parsed_text = remap_metaaddress(parsed_text)

        # Dump as JSON
        print(yaml.dump(parsed_text))
        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(ModelToJsonCommand())
