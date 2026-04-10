#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import IO

import click
import yaml

from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.utils.pipeline import get_pipeline_description


@click.command(hidden=True)
@click.option("-o", "--output", type=click.File("w"), default="-")
@pass_context
def dump_pipeline(ctx: ClickContext, output: IO[str]):
    pipeline_description = get_pipeline_description(ctx.obj.pipeline)
    yaml.safe_dump(pipeline_description, output)
