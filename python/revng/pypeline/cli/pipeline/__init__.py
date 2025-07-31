#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import click

from revng.pypeline.cli.utils import LazyGroup


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "run_pipe": "pypeline.cli.pipeline.run_pipe:run_pipe",
        "run_analysis": "pypeline.cli.pipeline.run_analysis:run_analysis",
    },
    help="Low-level pipeline commands (plumbing)",
)
def pipeline() -> None:
    pass
