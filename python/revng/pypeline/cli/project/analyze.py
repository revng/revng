#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import sys

import click

from revng.pypeline.cli.utils import build_arg_objects, build_help_text, compute_objects
from revng.pypeline.cli.utils import list_objects_for_container, normalize_whitespace
from revng.pypeline.cli.utils import storage_provider_factory
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.pipeline import AnalysisBinding, Pipeline
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils.registry import get_singleton

logger = logging.getLogger(__name__)


class AnalyzeGroup(click.Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type: type[Model] = get_singleton(Model)  # type: ignore[type-abstract]

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        pipeline = ctx.obj.get("pipeline")
        if pipeline is None:
            return base
        return base + sorted(pipeline.analyses.keys())

    def get_command(self, ctx, cmd_name):
        pipeline = ctx.obj.get("pipeline")
        if pipeline is None:
            return super().get_command(ctx, cmd_name)
        if cmd_name not in pipeline.analyses:
            return super().get_command(ctx, cmd_name)
        return self._build_analysis_command(
            analysis_name=cmd_name,
            pipeline=pipeline,
        )

    def _build_analysis_command(self, analysis_name: str, pipeline: Pipeline):
        """Dynamically create a command for running an analysis."""
        analysis_binding: AnalysisBinding = pipeline.analyses[analysis_name]

        if analysis_binding.analysis.__doc__:
            help_text = click.wrap_text(
                f"\n{normalize_whitespace(analysis_binding.analysis.__doc__)}"
            )
        else:
            help_text = f"Run the analysis: {analysis_name}"

        help_text = build_help_text(prologue=help_text, args=[])

        # Build the actual function that will be the command
        run_analysis_command = build_analysis_command(
            analysis_binding=analysis_binding,
            help_text=help_text,
            model_type=self.model_type,
            pipeline=pipeline,
        )

        config = getattr(
            analysis_binding.analysis,
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
        for container_decl in analysis_binding.bindings:
            run_analysis_command = build_arg_objects(container_decl)(run_analysis_command)

        return run_analysis_command


def build_analysis_command(
    analysis_binding: AnalysisBinding,
    help_text: str,
    model_type: type[Model],
    pipeline: Pipeline,
):
    analysis_name: str = analysis_binding.analysis.name

    @click.command(name=analysis_name, help=help_text)
    @click.argument(
        "model",
        type=click.Path(exists=True, dir_okay=False, readable=True),
        required=True,
    )
    @click.option(
        "--list",
        type=bool,
        is_flag=True,
        default=False,
        help="List the available objects for each argument.",
    )
    def run_analysis_command(
        model: str,
        configuration: str,
        **kwargs,
    ) -> None:
        logger.debug('Running analysis: "%s"', analysis_name)
        logger.debug('configuration: "%s"', configuration)
        logger.debug('model: "%s"', model)
        logger.debug('and kwargs: "%s"', kwargs)

        # Load the model
        storage_provider = storage_provider_factory(model_path=model)
        loaded_model = model_type.deserialize(storage_provider.get_model())

        logger.debug('Model loaded: "%s"', loaded_model)

        if kwargs["list"]:
            # If the user requested to list the available objects, we print them
            # and exit
            for container_decl in analysis_binding.bindings:
                list_objects_for_container(
                    model=ReadOnlyModel(loaded_model),
                    arg_name=container_decl.name,
                    kind=container_decl.container_type.kind,
                )
                # Space between containers
                print()
            return

        # Compute the requests for the incoming containers of the
        # analysis
        incoming = Requests()
        for container_decl in analysis_binding.bindings:
            incoming[container_decl] = compute_objects(
                model=ReadOnlyModel(loaded_model),
                arg_name=container_decl.name,
                kind=container_decl.container_type.kind,
                kwargs=kwargs,
            )

        # Finally, run the analysis
        new_model = pipeline.run_analysis(
            model=ReadOnlyModel(loaded_model),
            analysis_name=analysis_name,
            requests=incoming,
            analysis_configuration=configuration,
            pipeline_configuration={},
            storage_provider=storage_provider,
        )
        logger.debug("Analysis run completed")
        # Print on stdout the raw bytes of the modified model
        sys.stdout.buffer.write(new_model.serialize())

    return run_analysis_command


@click.group(
    cls=AnalyzeGroup,
    help="Run an analysis",
)
def analyze() -> None:
    pass
