#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
This is just a wrapper over `pype` that sets pipebox to the revng pipebox path.
The path is computed relatively to this file, so this should work regardless of
where revng is installed.
"""

import os
from pathlib import Path

import click

from revng.internal.support import cache_directory
from revng.pypeline.cli.project import project
from revng.pypeline.cli.utils import PypeGroup
from revng.pypeline.main import pype, run


@click.group(cls=PypeGroup)
def quick():
    """Quick commands (japanese toilet)"""
    # TODO


@click.command()
def init():
    """Initialize a new project."""
    # TODO


def patch_pype():
    """
    revng2 is based on `pype`, but we want to change some defaults to be revng specific,
    and we want to add some commands.
    """
    # Replace the name (needed for autocompletion and usage)
    pype.name = "revng2"
    pype.add_command(quick)
    # Replace the default for pipebox
    for param in pype.params:
        if param.name == "pipebox":
            param.default = os.environ.get("PIPEBOX", Path(__file__).parent.parent / "pipebox.py")

    # Add `init` to project subcommand
    project.add_command(init)
    # Change the default for pipeline
    for param in project.params:
        if param.name == "pipeline":
            param.default = os.environ.get(
                "PIPELINE", Path(__file__).parent.parent / "pipeline.yml"
            )
        elif param.name == "cache_dir":
            param.default = cache_directory()


def main():
    """Entry point for revng2."""
    patch_pype()
    run()


if __name__ == "__main__":
    main()
