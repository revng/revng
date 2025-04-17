#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging

import idb
import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.revng import run_revng_command
from revng.internal.cli.support import temporary_file_gen
from revng.model import YamlDumper  # type: ignore

from .idb_converter import IDBConverter


class ImportIDBCommand(Command):
    def __init__(self):
        super().__init__(
            ("model", "import", "idb"), "Extract a rev.ng model from an IDB/I64 database"
        )

    def register_arguments(self, parser):
        parser.add_argument("idb", help="Path to the IDB/I64 file to import")
        parser.add_argument(
            "--output", "-o", default="/dev/stdout", help="Output filepath (default stdout)"
        )
        parser.add_argument(
            "--base", default=0x0, help="base address where dynamic objects should be loaded"
        )

    def run(self, options: Options):
        # Suppress warnings from python-idb
        logging.basicConfig(level=logging.ERROR)

        idb_path = options.parsed_args.idb
        output_path = options.parsed_args.output

        # NOTE: This should be used for PIC only.
        base_address = options.parsed_args.base

        with idb.from_file(idb_path) as db:
            idb_converter = IDBConverter(db, base_address, options.parsed_args.verbose)
            revng_model = idb_converter.get_model()

        yaml_model = yaml.dump(revng_model, Dumper=YamlDumper)
        temporary_file = temporary_file_gen("revng-import-idb-", options)
        with temporary_file(suffix=".yml") as model_file:
            model_file.write(yaml_model)
            model_file.flush()

            # Fix the model and do the clean-up.
            run_revng_command(
                [
                    "model",
                    "opt",
                    "-fix-model",
                    model_file.name,
                    "-o",
                    output_path,
                ],
                options,
            )

        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(ImportIDBCommand())
