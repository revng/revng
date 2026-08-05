#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from tempfile import NamedTemporaryFile
from typing import Dict

import click
import yaml

from revng.internal.cli.common import ClickContext, CommandRegistry, WrappableCommand, pass_context
from revng.internal.cli.support import file_wrapper
from revng.model import YamlDumper  # type: ignore
from revng.model.migrations import migrate
from revng.support import log_error


def _write_result(model: Dict, input_: str | None, in_place: bool, output: str | None) -> None:
    if in_place:
        assert input_ is not None
        with open(input_, "w") as file:
            yaml.dump(model, file, Dumper=YamlDumper)
    else:
        with file_wrapper(output, "w") as file:
            yaml.dump(model, file, Dumper=YamlDumper)


@click.command(
    cls=WrappableCommand,
    name="migrate",
    help="Migrate model to the currently-supported schema version",
)
@click.argument("input_", metavar="[INPUT]", required=False)
@click.option(
    "-i",
    "--in-place",
    is_flag=True,
    help="If set, overwrites the model file and backs up the original model in a directory "
    "alongside the input file. Cannot be set along with --output. Can only be used when "
    "input is read from a file",
)
@click.option(
    "-o",
    "--output",
    help="The path to the output file, default to stdout. Cannot be set along with --in-place",
)
@pass_context
def model_migrate(ctx: ClickContext, input_: str | None, in_place: bool, output: str | None) -> int:
    if in_place and not input_:
        log_error("--in-place can only be used when input is read from a file")
        return 1

    if in_place and output:
        log_error("At most one of --in-place and --output can be specified")
        return 1

    with file_wrapper(input_, "r") as f:
        model = yaml.safe_load(f)

    migrate(model)

    with NamedTemporaryFile("w") as file:
        yaml.dump(model, file)
        model_verify_rc = ctx.obj.try_run(
            ["revng", "model", "opt", "--verify", "-o", os.devnull, file.name]
        )

    if model_verify_rc == 0:
        _write_result(model, input_, in_place, output)
        return 0
    else:
        log_error(
            "Migration result is invalid, please make sure the input model is valid and "
            "try again"
        )
        return 1


def setup(registry: CommandRegistry):
    registry.register(("model",), model_migrate)
