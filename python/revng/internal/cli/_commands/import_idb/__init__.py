#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging

import click
import idb
import yaml

from revng.internal.cli.common import ClickContext, CommandRegistry, WrappableCommand, pass_context
from revng.internal.cli.support import keep_temporaries_option, temporary_file_gen
from revng.model import YamlDumper  # type: ignore

from .idb_converter import IDBConverter


@click.command(
    cls=WrappableCommand,
    name="idb",
    help="Extract a rev.ng model from an IDB/I64 database",
)
@click.argument("idb_path", metavar="IDB")
@click.option("--output", "-o", default="/dev/stdout", help="Output filepath (default stdout)")
@click.option("--base", default=0x0, help="base address where dynamic objects should be loaded")
@keep_temporaries_option
@pass_context
def import_idb(
    ctx: ClickContext, idb_path: str, output: str, base: int, keep_temporaries: bool
) -> int:
    # Suppress warnings from python-idb
    logging.basicConfig(level=logging.ERROR)

    with idb.from_file(idb_path) as db:
        # NOTE: `base` should be used for PIC only.
        idb_converter = IDBConverter(db, base)
        revng_model = idb_converter.get_model()

    yaml_model = yaml.dump(revng_model, Dumper=YamlDumper)
    temporary_file = temporary_file_gen("revng-import-idb-", keep_temporaries)
    with temporary_file(suffix=".yml") as model_file:
        model_file.write(yaml_model)
        model_file.flush()

        # Fix the model and do the clean-up.
        return ctx.obj.try_run(
            ["revng", "model", "opt", "-fix-model", model_file.name, "-o", output]
        )


def setup(registry: CommandRegistry):
    registry.register(("model", "import"), import_idb)
