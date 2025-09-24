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
import psutil
from click.shell_completion import get_completion_class

from . import initialize_pypeline
from .cli.pipeline import pipeline
from .cli.project import project
from .cli.utils import EagerParsedPath

logger = logging.getLogger(__name__)


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
                'Pipebox file "%s" does not exist. Either set it using the '
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
        logger.error('Could not load module "%s" from "%s"', module_name, module_abspath)
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        if is_complete:
            return object()
        logger.error('Could not load module "%s" from "%s"', module_name, module_abspath)
        sys.exit(1)
    # Execute the module to load it
    spec.loader.exec_module(module)
    # Initialize the pypeline as we just imported the pipebox
    initialize_pypeline()
    return module


def get_current_root_name(ctx: click.Context) -> str:
    root = ctx.find_root()
    command = root.command
    assert command.name is not None, "Command name should not be None"
    return command.name


def detect_autocomplete(ctx: click.Context) -> bool:
    """Detect if we are in auto-complete mode."""
    return (
        "_{}_COMPLETE".format(get_current_root_name(ctx).upper()) in os.environ
        or "autocomplete" in sys.argv
    )


@click.group()
@click.option(
    "-C",
    "--directory",
    # This impacts `--pipebox` as well, so we need to change the cwd before
    # parsing any other argument
    type=EagerParsedPath(
        name="directory",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        parser=lambda path, _ctx: os.chdir(path) if path else None,
    ),
    help=(
        'Change the current working directory, equivalent to running "cd" before executing any'
        " command."
    ),
    expose_value=False,
)
@click.option(
    "--pipebox",
    # The pypebox needs to be imported before the arguments of the subcommands are parsed
    # so we use an eager option to import it as soon as possible
    type=EagerParsedPath(
        name="pipebox",
        # During auto-completion we don't want to fail if the file does not exist
        exists=False,
        parser=lambda path, ctx: import_pipebox(path, detect_autocomplete(ctx)),
    ),
    help=(
        'Path to the pipebox file. Defaults to the "PIPEBOX" environment '
        'variable, then "pipebox.py".'
    ),
    default=os.environ.get("PIPEBOX", "pipebox.py"),
    show_default=True,
    expose_value=False,
)
@click.pass_context
def pype(ctx):
    pass


pype.add_command(pipeline)
pype.add_command(project)


def detect_shell() -> str:
    """Detect the current shell."""
    try:
        # Get the parent process ID (PPID) of the current Python script
        ppid = os.getppid()
        # Get the process object from the PPID
        parent_process = psutil.Process(ppid)
        # The name of the executable of the parent process is our shell
        return parent_process.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Handle cases where the parent process might not exist or is inaccessible
        return Path(os.environ.get("SHELL", "bash")).name


@pype.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default=detect_shell(),
    help="Shell type",
)
@click.pass_context
def autocomplete(ctx, shell):
    """
    Generate shell completion script for the CLI.

    To temporary enable autocomplete run `eval "$(pype autocomplete)"`.
    To install them, depending on your shell, run:
    - `bash`: `pype autocomplete --shell bash > ~/.bash_completion.d/pype`
    - `zsh` : `pype autocomplete --shell zsh  > ~/.zsh/completions/_pype`
    - `fish`: `pype autocomplete --shell fish > ~/.config/fish/completions/pype.fish`
    """
    logger.info("Detected shell: %s", shell)

    # Get the root command
    prog_name = get_current_root_name(ctx)
    logger.debug("Program name: %s", prog_name)
    complete_var = f"_{prog_name.upper()}_COMPLETE"
    logger.debug("Complete variable: %s", complete_var)

    # Requires Click 8.0+ for shell_complete
    # Create a completion context
    completion_cls = get_completion_class(shell)
    if completion_cls is None:
        click.echo(f'Shell "{shell}" is not supported', err=True)
        return

    completion = completion_cls(
        cli=ctx.find_root().command,
        ctx_args={},
        prog_name=prog_name,
        complete_var=complete_var,
    )
    click.echo(completion.source())


def main(args: Sequence[str]) -> None:
    # pylint: disable=E1120 no-value-for-parameter
    # This is ok as click will pass the pipebox argument automatically
    pype(args=args)


def run():
    """Run the pipeline from the command line using the shell environment."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
