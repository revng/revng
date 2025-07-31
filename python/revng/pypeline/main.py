#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import logging
import os
import sys
from typing import Sequence

import click

from revng.pypeline.cli.utils import EagerParsedPath, LazyGroup
from revng.pypeline.utils.import_pipebox import import_pipebox

logger = logging.getLogger("pypeline")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stderr))


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "pipeline": "pypeline.cli.pipeline:pipeline",
        "project": "pypeline.cli.project:project",
    },
)
@click.option(
    "--pipebox",
    type=EagerParsedPath(
        name="pipebox",
        parser=lambda path: import_pipebox(path, "_PYPE_COMPLETE" in os.environ),
    ),
    help=(
        "Path to the pipebox file. Defaults to the `PIPEBOX` environment "
        "variable, then 'pipebox.py'."
    ),
    default=os.environ.get("PIPEBOX", "pipebox.py"),
    expose_value=False,
)
def cli():
    pass


def main(args: Sequence[str]) -> None:
    # pylint: disable=E1120 no-value-for-parameter
    # This is ok as click will pass the pipebox argument automatically
    cli(args=args)


def run():
    """Run the pipeline from the command line using the shell environment."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
