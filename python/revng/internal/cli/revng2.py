#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
This is just a wrapper over `pype` that sets pipebox to the revng pipebox path.
The path is computed relatively to this file, so this should work regardless of
where revng is installed.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import click

from revng.pypeline.cli.utils import LazyGroup
from revng.pypeline.main import import_pipebox

logger = logging.getLogger("revng2")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stderr))


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "pipeline": "revng.pypeline.cli.pipeline:pipeline",
        "project": "revng.pypeline.cli.project:project",
    },
)
def cli():
    pass


def main(args: Sequence[str]) -> None:
    # This should resolve to the full path of revng/internal/pipebox.py
    pipebox_path = Path(__file__).parent.parent / "pipebox.py"
    import_pipebox(str(pipebox_path), "_REVNG2_COMPLETE" in os.environ)
    # pylint: disable=E1120 no-value-for-parameter
    # This is ok as click will pass the pipebox argument automatically
    cli(args=args)


def run():
    """Run the pipeline from the command line using the shell environment."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
