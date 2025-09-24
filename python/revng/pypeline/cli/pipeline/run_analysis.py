#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import sys

import click

from revng.pypeline.analysis import Analysis
from revng.pypeline.cli.utils import build_arg_objects, build_help_text, compute_objects
from revng.pypeline.cli.utils import list_objects_option, normalize_whitespace
from revng.pypeline.container import ContainerDeclaration, load_container
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.task.task import TaskArgument, TaskArgumentAccess
from revng.pypeline.utils.registry import get_registry, get_singleton

logger = logging.getLogger(__name__)


class RunAnalysisGroup(click.Group):
    """We need to create a custom command for each analysis we loaded from the registry.
    Since we already have to generate the code dynamically, we do it lazily so
    we generate only the commands that are requested."""

    @property
    def registry(self) -> dict[str, type[Analysis]]:
        return get_registry(Analysis)  # type: ignore[type-abstract]

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        return base + sorted(self.registry.keys())

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.registry:
            return self._build_analysis_command(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _build_analysis_command(self, analysis_name: str):
        """Dynamically create a command for running an analysis."""
        analysis_type: type[Analysis] = self.registry[analysis_name]

        if analysis_type.__doc__:
            help_text = click.wrap_text(f"\n{normalize_whitespace(analysis_type.__doc__)}")
        else:
            help_text = f"Run the analysis: {analysis_name}"

        help_text = build_help_text(
            prologue=help_text,
            args=[
                TaskArgument(
                    name=container_type.__name__,
                    container_type=container_type,
                    access=TaskArgumentAccess.READ,
                    help_text=normalize_whitespace(container_type.__doc__ or ""),
                )
                for container_type in analysis_type.signature()
            ],
        )

        # Build the actual function that will be the command
        run_analysis_command = build_run_analysis_command(
            analysis_name=analysis_name,
            help_text=help_text,
            analysis_type=analysis_type,
            model_type=get_singleton(Model),  # type: ignore[type-abstract]
        )

        config = getattr(
            analysis_type,
            "configuration_help",
            f'Configuration for the analysis "{analysis_name}".',
        )
        if config is not None:
            run_analysis_command = click.option(
                "-c",
                "--configuration",
                type=str,
                default="",
                help=normalize_whitespace(config),
            )(run_analysis_command)

        # For each argument, call the `click.argument` decorator to dynamically add
        # them to the command
        for arg in analysis_type.signature():
            run_analysis_command = click.argument(
                arg.__name__,
                type=click.Path(exists=True, dir_okay=False, readable=True),
            )(run_analysis_command)
            run_analysis_command = build_arg_objects(
                ContainerDeclaration(
                    name=arg.__name__,
                    container_type=arg,
                )
            )(run_analysis_command)

        return run_analysis_command


def build_run_analysis_command(
    analysis_name: str,
    help_text: str,
    analysis_type: type[Analysis],
    model_type: type[Model],
):
    @click.command(name=analysis_name, help=help_text)
    @click.argument(
        "model",
        type=click.Path(exists=True, dir_okay=False, readable=True),
        required=True,
    )
    @list_objects_option
    def run_analysis_command(
        model: str,
        configuration: str,
        **kwargs,
    ) -> None:
        logger.debug('Running analysis: "%s"', analysis_name)
        logger.debug('configuration: "%s"', configuration)
        logger.debug('model: "%s"', model)
        logger.debug('and kwargs: "%s"', kwargs)

        analysis = analysis_type(
            name=analysis_name,
        )
        # Load the model
        loaded_model: Model = model_type()
        with open(model, "rb") as model_file:
            loaded_model = model_type.deserialize(model_file.read())

        logger.debug('Model loaded: "%s"', loaded_model)

        # Load the containers with args form the command line
        containers = []
        for arg in analysis.signature():
            arg_name = arg.__name__
            path = kwargs[arg_name]
            container = load_container(arg, path)
            logger.debug(
                'Loaded container from "%s" for argument "%s": "%r"', path, arg_name, container
            )
            containers.append(container)

        # Compute the requests for the incoming containers of the
        # analysis
        incoming: list[ObjectSet] = []
        for arg in analysis.signature():
            # If the argument is writable, we need to request the objects
            incoming.append(
                compute_objects(
                    model=ReadOnlyModel(loaded_model),
                    arg_name=arg.__name__,
                    kind=arg.kind,
                    kwargs=kwargs,
                )
            )

        # Finally, run the analysis
        analysis.run(
            model=loaded_model,
            containers=containers,
            incoming=incoming,
            configuration=configuration,
        )
        logger.debug("Analysis run completed")
        # Print on stdout the raw bytes of the modified model
        sys.stdout.buffer.write(loaded_model.serialize())

    return run_analysis_command


@click.group(
    cls=RunAnalysisGroup,
    help="Run an analysis",
)
def run_analysis() -> None:
    pass
