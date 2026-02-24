#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path

import click
from click.core import ParameterSource

from revng.pypeline.cli.context import ClickContext
from revng.pypeline.cli.utils import EagerParsedPath
from revng.pypeline.container import ContainerFormat
from revng.pypeline.runner_context import RunnerContext

# Options that are common to multiple commands
project_id_option = click.option(
    "--project-id",
    type=str,
    help="Project id to use for the storage provider.",
    envvar="PYPELINE_PROJECT_ID",
    show_default=True,
)

list_objects_option = click.option(
    "--list",
    type=bool,
    is_flag=True,
    default=False,
    help="List the available objects for each argument.",
)

token_option = click.option(
    "--token",
    type=str,
    required=False,
    help="The token to pass to the storage provider.",
)


_FORMAT_VARIABLE = "container_format"


def handle_format_option(ctx, param, value):
    """
    Handles mutual exclusivity, converts strings to Enums,
    and manages precedence between flags and defaults.
    """
    # Handle shortcuts the value is boolean here
    if param.name in [f.value for f in ContainerFormat]:
        if not value:
            return None

        if ctx.params.get(_FORMAT_VARIABLE) is not None:
            raise click.BadOptionUsage(
                param.name, f"Mutually exclusive: --{param.name} cannot be used with other formats."
            )

        ctx.params[_FORMAT_VARIABLE] = ContainerFormat(param.name)
        return None

    # Handle --format, value is a string passed from click.Choice or default

    # Check if a shortcut already populated the format
    existing_format = ctx.params.get(_FORMAT_VARIABLE)
    if existing_format is not None:
        # If the user EXPLICITLY typed --format AND used a flag -> Error
        if ctx.get_parameter_source(param.name) != ParameterSource.DEFAULT:
            raise click.BadOptionUsage(
                param.name,
                "Mutually exclusive: Cannot specify "
                f"--format when --{existing_format.value} is used.",
            )
        # If --format is just running its default 'yaml', let the Flag win
        return existing_format

    # If no flag was used, convert the string value to Enum and return
    return ContainerFormat(value)


def container_format_options(func):
    func = click.option(
        "--format",
        _FORMAT_VARIABLE,
        type=click.Choice([x.value for x in ContainerFormat]),
        default=ContainerFormat.YAML.value,
        show_default=True,
        callback=handle_format_option,
        help="Format to use for the output container, either on stdout or in the result path.",
    )(func)
    for member in ContainerFormat:
        func = click.option(
            f"--{member.value}",
            is_flag=True,
            expose_value=False,
            help=f"Shortcut for --format={member.value}.",
            callback=handle_format_option,
        )(func)
    return func


def _parse_debug_option(path: str, ctx: ClickContext):
    return RunnerContext(True, ctx.obj.pipebox.argv_hook, Path(path))


debug_option = click.option(
    "--debug",
    "runner_context",
    type=EagerParsedPath(
        name="runner_context",
        parser=_parse_debug_option,
        default=RunnerContext(),
        dir_okay=True,
        file_okay=False,
        exists=False,
    ),
    help=(
        """
        Run the command in debug mode with the specified directory (it will be
        created if missing). Where possible the pipes and analyses will be run
        as subcommands via `run-pipe` and `run-analysis` with input and output
        files in subdirectories in the specified directory.
        """
    ),
    default=EagerParsedPath.DEFAULT,
    show_default=False,
)
