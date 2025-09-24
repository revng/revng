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

import click
from xdg import xdg_cache_home

from revng.pypeline.cli.project import project
from revng.pypeline.main import pype

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


@click.group()
def quick():
    """Quick commands (japanese toilet)"""
    # WIP


@quick.command()
def init():
    """Initialize a new project."""
    # WIP


def main():
    # Replace the name (needed for autocompletion and usage)
    pype.name = "revng2"
    pype.lazy_subcommands["quick"] = "revng.internal.cli.revng2:quick"
    # Replace the default for pipebox
    for param in pype.params:
        if param.name == "pipebox":
            param.default = os.environ.get("PIPEBOX", Path(__file__).parent.parent / "pipebox.py")

    # Add `init` to project subcommand
    project.lazy_subcommands["init"] = "revng.internal.cli.revng2:init"
    # Change the default for pipeline
    for param in project.params:
        if param.name == "pipeline":
            param.default = os.environ.get(
                "PIPELINE", Path(__file__).parent.parent / "pipeline.yml"
            )
        elif param.name == "cache_dir":
            param.default = os.environ.get(
                "CACHE_DIR",
                xdg_cache_home() / "revng",
            )

    pype()


if __name__ == "__main__":
    main()()
