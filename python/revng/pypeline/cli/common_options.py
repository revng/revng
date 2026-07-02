#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from enum import Enum
from pathlib import Path
from typing import Generator, cast

import click
from click_option_group import GroupedOption, OptionGroup

from revng.pypeline.analysis import Analysis, AnalysisList
from revng.pypeline.cli.context import ClickContext
from revng.pypeline.cli.utils import EagerParsedPath, detect_autocomplete, normalize_flag
from revng.pypeline.container import ContainerFormat
from revng.pypeline.pipeline import AnalysisBinding, Pipeline
from revng.pypeline.pipeline_node import PipelineNode
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.utils.registry import get_registry

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


def make_format_callback(default: ContainerFormat):
    """
    Build the callback that handles mutual exclusivity, converts strings to
    Enums, and falls back to the given default when nothing is specified.
    """

    def handle_format_option(ctx, param, value):
        # Handle shortcuts the value is boolean here
        if param.name in [f.value for f in ContainerFormat]:
            if not value:
                return None

            if ctx.params.get(_FORMAT_VARIABLE) is not None:
                raise click.BadOptionUsage(
                    param.name,
                    f"Mutually exclusive: --{param.name} cannot be used with other formats.",
                )

            ctx.params[_FORMAT_VARIABLE] = ContainerFormat(param.name)
            return None

        # Handle --format, value is a string passed from click.Choice or default

        # Check if a shortcut already populated the format
        existing_format = ctx.params.get(_FORMAT_VARIABLE)
        if existing_format is not None:
            # If the user EXPLICITLY typed --format AND used a flag -> Error
            if value is not None:
                raise click.BadOptionUsage(
                    param.name,
                    "Mutually exclusive: Cannot specify "
                    f"--format when --{existing_format.value} is used.",
                )
            # If --format is just running its default, let the Flag win
            return existing_format

        if value is not None:
            # If no flag was used, convert the string value to Enum and return
            return ContainerFormat(value)
        else:
            return default

    return handle_format_option


def container_format_options(default: ContainerFormat):
    callback = make_format_callback(default)

    def decorator(func):
        func = click.option(
            "--format",
            _FORMAT_VARIABLE,
            type=click.Choice([x.value for x in ContainerFormat]),
            callback=callback,
            help=(
                "Format to use for the output container, either on stdout or in "
                f"the result path.  [default: {default.value}]"
            ),
        )(func)
        for member in ContainerFormat:
            # No shortcut for AUTO: it is the default and "--auto" is too generic.
            if member is ContainerFormat.AUTO:
                continue
            func = click.option(
                f"--{member.value}",
                is_flag=True,
                expose_value=False,
                help=f"Shortcut for --format={member.value}.",
                callback=callback,
            )(func)
        return func

    return decorator


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


def _show_full_help(ctx: ClickContext, param, value: bool):
    if not value:
        return

    ctx.obj.show_hidden = True
    click.echo(ctx.get_help())
    ctx.exit()


full_help = click.option(
    "--help-full",
    is_flag=True,
    callback=_show_full_help,
    is_eager=True,
    expose_value=False,
    help="Show help with hidden options and exit.",
)


class _ConfigurationOptionType(click.ParamType):
    def __init__(self, key: type[Pipe] | type[Analysis]):
        self.name = f"{normalize_flag(key.name)}-configuration"
        self.key = key

    def convert(self, value, param, ctx: ClickContext):  # type: ignore
        if issubclass(self.key, Analysis):
            for binding in ctx.obj.pipeline.analyses.values():
                if isinstance(binding.analysis, self.key):
                    ctx.obj.configuration[binding.analysis] = value
        else:
            for node in ctx.obj.pipeline.walk_pipeline():
                if isinstance(node.task, self.key):
                    ctx.obj.configuration[node.task] = value


class _HidableOption(GroupedOption):
    @property
    def hidden(self):
        ctx: ClickContext | None = cast(ClickContext | None, click.get_current_context(silent=True))
        if detect_autocomplete(ctx):
            return False
        return False if ctx is None else not ctx.obj.show_hidden

    @hidden.setter
    def hidden(self, value):
        pass


class AllAnalysesOption(Enum):
    ALL_ANALYSES = 0


ConfigTarget = PipelineNode | AllAnalysesOption | AnalysisBinding | AnalysisList
ComponentSet = set[type[Pipe] | type[Analysis]]


def _flatten(gen: Generator[ComponentSet, None, None]) -> ComponentSet:
    result: ComponentSet = set()
    for element in gen:
        result.update(element)
    return result


def _get_config_components(pipeline: Pipeline, target: ConfigTarget) -> ComponentSet:
    result: ComponentSet = set()
    if isinstance(target, PipelineNode):
        nodes: list[PipelineNode] = [target]
        while len(nodes) > 0:
            node = nodes.pop(0)
            if isinstance(node.task, Pipe):
                result.add(node.task.__class__)
            nodes.extend(node.predecessors)
        return result
    elif isinstance(target, AnalysisBinding):
        return _get_config_components(pipeline, target.node)
    elif isinstance(target, AnalysisList):
        for analysis_name in target.analyses:
            binding = pipeline.analyses[analysis_name]
            result.add(binding.analysis.__class__)
            result.update(_get_config_components(pipeline, binding.node))
        return result
    elif target is AllAnalysesOption.ALL_ANALYSES:
        for analysis_type in get_registry(Analysis).values():  # type: ignore[type-abstract]
            result.add(analysis_type)
            for analysis_binding in pipeline.analyses.values():
                if isinstance(analysis_binding.analysis, analysis_type):
                    result.update(_get_config_components(pipeline, analysis_binding.node))
        return result
    else:
        raise ValueError


def add_pipeline_config_options(pipeline: Pipeline, *targets: ConfigTarget):
    def decorator(func):
        group = OptionGroup("Pipe/Analysis configuration options")
        values = _flatten(_get_config_components(pipeline, t) for t in targets)
        for type_ in sorted(values, key=lambda x: x.name):
            type_string = "pipe" if issubclass(type_, Pipe) else "analysis"
            func = group.option(
                f"--{normalize_flag(type_.name)}-configuration",
                type=_ConfigurationOptionType(type_),
                expose_value=False,
                cls=_HidableOption,
                metavar="CONFIGURATION",
                help=f"Configuration for the {type_.name} {type_string}",
            )(func)

        func = full_help(func)

        return func

    return decorator
