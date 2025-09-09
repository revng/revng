#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import click

from . import initialize_pypeline
from .cli.utils import EagerParsedPath, LazyGroup

logger = logging.getLogger("pypeline")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stderr))


def import_pipebox(module_path: str, is_complete: bool) -> object:
    """Import a module from a path. This is used to import the pipebox file.
    Args:
        env (dict[str, str]): The environment variables.
        module_path (str): The path to the module.
        is_complete (bool): If True, raise an error if the module is not found.
    """
    # Absolute path to the module
    module_abspath = Path(module_path).resolve()
    if not module_abspath.exists():
        # This is a small trick to allow generating the module auto-complete
        # without having a pipebox file
        if is_complete:
            return object()
        logger.error(
            (
                "Pipebox file `%s` does not exist. Either set it using the "
                "PIPEBOX env var, or pass the --pipebox option."
            ),
            module_abspath,
        )
        sys.exit(1)
    # We guess that the module name is the file name without the extension
    module_name: str = module_abspath.stem
    # Dynamic import of the pipebox module
    spec = importlib.util.spec_from_file_location(module_name, str(module_abspath))
    if spec is None:
        if is_complete:
            return object()
        logger.error("Could not load module `%s` from `%s`", module_name, module_abspath)
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        if is_complete:
            return object()
        logger.error("Could not load module `%s` from `%s`", module_name, module_abspath)
        sys.exit(1)
    # Execute the module to load it
    spec.loader.exec_module(module)
    # Initialize the pypeline as we just imported the pipebox
    initialize_pypeline()
    return module


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "pipeline": "revng.pypeline.cli.pipeline:pipeline",
        "project": "revng.pypeline.cli.project:project",
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
