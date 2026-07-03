#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import click
import psutil
from click.shell_completion import get_completion_class

from revng.pypeline.cli.context import ClickContext, ContextObject, pass_context
from revng.pypeline.utils import PypelineException

from . import initialize_pypeline
from .cli.pipeline import pipeline
from .cli.project import project
from .cli.rss import rss
from .cli.utils import EagerParsedPath, detect_autocomplete, get_root_command_name
from .utils.logger import pypeline_logger


class WideHelpFormatter(click.HelpFormatter):
    def __init__(self, *args, **kwargs):
        # Click by default clamps the terminal width to 78 characters (80 - 2),
        # force the actual terminal width minus 2 to have some margin
        terminal_width = shutil.get_terminal_size()[0] - 2
        # Now initialize the formatter with the computed value
        super().__init__(width=terminal_width, max_width=terminal_width)
        # Maintain the proportion of 30 columns over a 80 column screen
        # This value is used to set the maximum column size of the flags column
        # in the help
        self.col_max = int(terminal_width * 0.375)

    def write_dl(
        self,
        rows: Iterable[tuple[str, str]],
        col_max: int | None = None,
        col_spacing: int = 2,
    ) -> None:
        if col_max is None:
            col_max = self.col_max
        super().write_dl(rows, col_max, col_spacing)


click.Context.formatter_class = WideHelpFormatter


def import_pipebox(module_path: str, is_autocomplete: bool) -> object:
    """Import a module from a path. This is used to import the pipebox file.
    Args:
        env (dict[str, str]): The environment variables.
        module_path (str): The path to the module.
        is_autocomplete (bool): If True, do not raise an error if the pipebox
            is not found, since when autocompleting the pipebox path might be
            missing or incomplete.
    """
    # Absolute path to the module
    module_abspath = Path(module_path).resolve()
    if not module_abspath.exists():
        # This is a small trick to allow generating the module auto-complete
        # without having a pipebox file
        if is_autocomplete:
            return object()
        pypeline_logger.log(
            f'Pipebox file "{module_abspath}" does not exist. Either set it using the '
            "PYPELINE_PIPEBOX env var, or pass the --pipebox option."
        )
        sys.exit(1)
    # We guess that the module name is the file name without the extension
    module_name: str = module_abspath.stem
    # Dynamic import of the pipebox module
    spec = importlib.util.spec_from_file_location(module_name, str(module_abspath))
    if spec is None:
        if is_autocomplete:
            return object()
        pypeline_logger.log(f'Could not load module "{module_name}" from "{module_abspath}"')
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        if is_autocomplete:
            return object()
        pypeline_logger.log(f'Could not load module "{module_name}" from "{module_abspath}"')
        sys.exit(1)
    # Execute the module to load it
    spec.loader.exec_module(module)
    # Initialize the pypeline as we just imported the pipebox
    initialize_pypeline()
    # If auto-completing also initialize the pipebox with empty arguments here
    # instead of in the body of pype
    if is_autocomplete:
        call_pipebox_initialize(module, module_path, [])
    return module


def call_pipebox_initialize(pipebox, pipebox_path: Path | str, args: list[str]):
    # Get its initialize function
    pipebox_initialize = getattr(pipebox, "initialize", None)
    if pipebox_initialize is None:
        pypeline_logger.log(
            f'Pipebox file "{pipebox_path!s}" does not have an "initialize" function.'
            " This is required to setup the pypeline."
        )
        sys.exit(1)

    # Call the initialize
    pipebox_initialize(args)


def parse_pipebox(path: str, ctx: ClickContext):
    pipebox = import_pipebox(path, detect_autocomplete(ctx))
    ctx.obj.pipebox = pipebox
    return pipebox


def parse_base_directory(path: str, ctx: ClickContext):
    if path:
        ctx.obj.base_directory = Path(path)


@click.group
@click.option(
    "-C",
    "--directory",
    type=EagerParsedPath(
        name="directory",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        parser=parse_base_directory,
    ),
    help="Run the command as it was started in the specified directory",
    default="",
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
        parser=parse_pipebox,
    ),
    help=(
        'Path to the pipebox file. Defaults to the "PYPELINE_PIPEBOX" '
        'environment variable, then "pipebox.py".'
    ),
    default="pipebox.py",
    envvar="PYPELINE_PIPEBOX",
    show_default=True,
    expose_value=False,
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging for the pypeline related code.",
)
@pass_context
def pype(ctx: ClickContext, verbose: bool) -> None:
    # Enable debug logging for pypeline if requested
    if verbose:
        pypeline_logger.debug = True

    # Avoid initializing the pipebox if we are in auto-complete mode
    if detect_autocomplete(ctx):
        return

    # Initialize the pipebox
    call_pipebox_initialize(ctx.obj.pipebox, ctx.obj.pipebox_path, ctx.obj.pipebox_args)


pype.add_command(pipeline)
pype.add_command(project)
pype.add_command(rss)


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
    pypeline_logger.log(f'Detected shell: "{shell}"')

    # Get the root command
    prog_name = get_root_command_name(ctx)
    pypeline_logger.debug_log(f'Program name: "{prog_name}"')
    complete_var = f"_{prog_name.upper()}_COMPLETE"
    pypeline_logger.debug_log(f'Complete variable: "{complete_var}"')

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
    # Divide click's argument from pipebox's arguments
    if "--" in args:
        position = args.index("--")
        click_args = args[:position]
        pipebox_args = args[position + 1 :]
    else:
        click_args = args
        pipebox_args = []

    # This is ok as click will pass the pipebox argument automatically
    try:
        exit_code = pype.main(
            args=click_args,
            obj=ContextObject.make(pipebox_args=pipebox_args),
            standalone_mode=False,
        )
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except PypelineException as e:
        pypeline_logger.log(str(e))
        sys.exit(1)

    sys.exit(exit_code)


def run():
    """Run the pipeline from the command line using the shell environment."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
