#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click

from .run_analysis import run_analysis
from .run_pipe import run_pipe


@click.group(
    help="Low-level pipeline commands (plumbing)",
)
def pipeline() -> None:
    pass


pipeline.add_command(run_pipe)
pipeline.add_command(run_analysis)
