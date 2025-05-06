#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import os
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.revng import run_revng_command
from revng.model import YamlDumper  # type: ignore
from revng.model.migrations import migrate
from revng.support import log_error


class ModelMigrateCommand(Command):
    def __init__(self):
        super().__init__(
            ("model", "migrate"),
            "Migrate model to the currently-supported schema version",
        )

    def register_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input",
            type=str,
            help="The path to the model file, defaults to stdin",
            nargs="?",
            default=None,
        )
        parser.add_argument(
            "-i",
            "--in-place",
            action="store_true",
            help="If set, overwrites the model file and backs up the original model in a directory "
            + "alongside the input file. Cannot be set along with --output. Can only be used when "
            + "input is read from a file",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            help="The path to the output file, default to stdout. Cannot be set along with "
            + "--in-place",
        )

    def _read_input(self, args: Any) -> Dict:
        if not args.input:
            return yaml.safe_load(sys.stdin)
        else:
            with open(args.input) as file:
                return yaml.safe_load(file)

    def _validate_schema(self, model: Dict, options: Options) -> int:
        with NamedTemporaryFile("w") as file:
            yaml.dump(model, file)
            args = ["model", "opt", "--verify", "-o", os.devnull, file.name]
            exit_code = run_revng_command(args, options)
            return exit_code

    def _write_result(self, model: Dict, args: Any) -> None:
        if args.in_place:
            with open(args.input, "w") as file:
                yaml.dump(model, file, Dumper=YamlDumper)
        elif args.output:
            with open(args.output, "w") as file:
                yaml.dump(model, file, Dumper=YamlDumper)
        else:
            yaml.dump(model, sys.stdout, Dumper=YamlDumper)

    def run(self, options: Options) -> int:
        args = options.parsed_args

        if args.in_place and not args.input:
            log_error("--in-place can only be used when input is read from a file")
            return 1

        if args.in_place and args.output:
            log_error("At most one of --in-place and --output can be specified")
            return 1

        model = self._read_input(args)

        migrate(model)

        if self._validate_schema(model, options) == 0:
            self._write_result(model, args)
            return 0
        else:
            log_error(
                "Migration result is invalid, please make sure the input model is valid and "
                + "try again"
            )
            return 1


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(ModelMigrateCommand())
