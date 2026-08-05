#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click

from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.utils import load_pipebox

from .run_analysis import run_analysis
from .run_pipe import run_pipe


@click.group(help="Low-level pipeline commands (plumbing)")
@pass_context
def pipeline(ctx: ClickContext) -> None:
    load_pipebox(ctx)


pipeline.add_command(run_pipe)
pipeline.add_command(run_analysis)
