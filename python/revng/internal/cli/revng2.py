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
from click.shell_completion import get_completion_class

from revng.pypeline.cli.utils import EagerParsedPath, LazyGroup
from revng.pypeline.main import import_pipebox

logger = logging.getLogger("revng2")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@click.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default=Path(os.environ.get("SHELL", "bash")).name,
    help="Shell type",
)
@click.pass_context
def autocomplete(ctx, shell):
    """
    Generate shell completion script for the CLI.

    To temporary enable autocomplete run `eval "$(revng2 autocomplete)"`.
    To install them, depending on your shell, run:
    - `bash`: `revng2 autocomplete --shell bash > ~/.bash_completion.d/revng2`
    - `zsh`: `revng2 autocomplete --shell zsh   > ~/.zsh/completions/_revng2`
    - `fish`: `revng2 autocomplete --shell fish > ~/.config/fish/completions/revng2.fish`
    """
    logger.info("Detected shell: %s", shell)

    # Get the root command
    root_cmd = ctx.find_root().command
    prog_name = "revng2"
    complete_var = "_REVNG2_COMPLETE"

    # Requires Click 8.0+ for shell_complete
    # Create a completion context
    completion_cls = get_completion_class(shell)
    if completion_cls is None:
        click.echo(f"Shell '{shell}' is not supported", err=True)
        return

    completion = completion_cls(
        cli=root_cmd,
        ctx_args={},
        prog_name=prog_name,
        complete_var=complete_var,
    )
    click.echo(completion.source())


@click.group(
    cls=LazyGroup,
    lazy_subcommands={
        "pipeline": "revng.pypeline.cli.pipeline:pipeline",
        "project": "revng.pypeline.cli.project:project",
        "daemon": "revng.internal.daemon2.main:run_server",
        "autocomplete": "revng.internal.cli.revng2:autocomplete",
    },
)
@click.option(
    "--pipebox",
    type=EagerParsedPath(
        name="pipebox",
        parser=lambda path: import_pipebox(str(path), "_REVNG2_COMPLETE" in os.environ),
    ),
    help=(
        "Path to the pipebox file. Defaults to the `PIPEBOX` environment "
        "variable, then 'pipebox.py'."
    ),
    # This should resolve to the full path of revng/internal/pipebox.py
    default=os.environ.get("PIPEBOX", Path(__file__).parent.parent / "pipebox.py"),
    expose_value=False,
)
@click.pass_context
def cli(ctx):
    if ctx.obj is None:
        ctx.obj = {}


def main(args: Sequence[str]) -> None:
    # pylint: disable=E1120 no-value-for-parameter
    # we have to pass pipebox thought obj instead of letting it be computed by
    # the `cli` function, because we need it also during auto-complete and
    # functions are not called during auto-complete.
    cli(args=args)


def run():
    """Run the pipeline from the command line using the shell environment."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
