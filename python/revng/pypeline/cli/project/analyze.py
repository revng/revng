#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
from typing import IO

import click
import yaml

from revng.pypeline.cli.backend import BackendFactory, BackendFeature, backend_factory_for
from revng.pypeline.cli.common_options import add_pipeline_config_options, debug_option
from revng.pypeline.cli.common_options import list_objects_option, project_id_option
from revng.pypeline.cli.common_options import read_configuration_file, token_option
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.utils import build_arg_objects, build_help_text, compute_objects
from revng.pypeline.cli.utils import list_objects_for_container, normalize_whitespace
from revng.pypeline.cli.wrappers import WrappablePypeCommand, exec_wrapper_if_needed
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.pipeline import AnalysisBinding, AnalysisList, ContainerDeclaration, Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.storage_provider import LockType
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils.logger import pypeline_logger

output_option = click.option(
    "-o",
    "output_file",
    type=click.File("wb"),
    help=(
        "Path to write the changed model to, if not specified, the "
        "result will be printed to stdout."
    ),
    default="-",
)

invalidation_option = click.option(
    "--invalidations",
    "invalidations_file",
    type=click.File("w"),
    help="Write invalidation data to the specified file",
)


async def async_part_of_command(
    backend_factory: BackendFactory,
    project_id: str,
    token: str,
    pipeline: Pipeline,
    runner_context: RunnerContext,
    analysis: str | AnalysisList,
    configuration: PipelineConfiguration,
    container_decls: tuple[ContainerDeclaration, ...],
    output_file: IO[bytes],
    invalidations_file: IO[bytes] | None,
    kwargs,
):
    """Since the backend session is an async context manager, the code that uses
    it has to be an async function.
    """
    async with backend_factory.session(
        lock_type=LockType.ANALYSIS,
        project_id=project_id,
        token=token,
        runner_context=runner_context,
    ) as session:
        loaded_model = session.get_model(configuration)[0]
        pypeline_logger.debug_log(f'Model loaded: "{loaded_model}"')

        if kwargs["list"]:
            # If the user requested to list the available objects, we print them
            # and exit
            for container_decl in container_decls:
                list_objects_for_container(
                    model=ReadOnlyModel(loaded_model),
                    arg_name=container_decl.name,
                    kind=container_decl.container_type.kind,
                )
                # Space between containers
                print()
            return

        if isinstance(analysis, str):
            # Compute the requests for the incoming containers of the
            # analysis
            incoming = Requests()
            for container_decl in container_decls:
                incoming[container_decl] = compute_objects(
                    model=ReadOnlyModel(loaded_model),
                    arg_name=container_decl.name,
                    kind=container_decl.container_type.kind,
                    kwargs=kwargs,
                )

            new_model, invalidated = session.run_analysis(
                ReadOnlyModel(loaded_model), analysis, incoming, configuration
            )
        else:
            new_model, invalidated = session.run_analysis_list(
                ReadOnlyModel(loaded_model), analysis, configuration
            )

        pypeline_logger.debug_log("Analysis run completed")
        # Print on the output_file the raw bytes of the modified model
        output_file.write(new_model.serialize())

        for container_location, object_ids in invalidated.items():
            serialized_ids = (object_id.serialize() for object_id in object_ids)
            pypeline_logger.debug_log(
                f"Invalidated {container_location}: [{', '.join(serialized_ids)}]"
            )

        if invalidations_file is not None:
            name_mapping = pipeline.savepoint_id_to_name
            data = []
            for container_location, object_ids in invalidated.items():
                objects = [obj.serialize() for obj in object_ids]
                data.append(
                    {
                        "savepoint": name_mapping[container_location.savepoint_id],
                        "container": container_location.container_id,
                        "configuration": container_location.configuration_id,
                        "objects": objects,
                    }
                )
            yaml.safe_dump(data, invalidations_file)


class AnalyzeGroup(click.Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx: ClickContext):  # type: ignore
        base = super().list_commands(ctx)
        pipeline = ctx.obj.pipeline
        return base + sorted(pipeline.analysis_lists.keys()) + sorted(pipeline.analyses.keys())

    def get_command(self, ctx: ClickContext, cmd_name):  # type: ignore
        pipeline = ctx.obj.pipeline
        if pipeline is None:
            return super().get_command(ctx, cmd_name)
        if cmd_name in pipeline.analyses:
            return self._build_analysis_command(
                analysis_name=cmd_name,
                pipeline=pipeline,
            )
        elif cmd_name in pipeline.analysis_lists:
            return self._build_analysis_list_command(
                analysis_list_name=cmd_name,
                pipeline=pipeline,
            )
        return super().get_command(ctx, cmd_name)

    def _build_analysis_command(self, analysis_name: str, pipeline: Pipeline):
        """Dynamically create a command for running an analysis."""
        analysis_binding: AnalysisBinding = pipeline.analyses[analysis_name]

        if analysis_binding.analysis.__doc__:
            help_text = click.wrap_text(
                f"\n{normalize_whitespace(analysis_binding.analysis.__doc__)}"
            )
        else:
            help_text = f"Run the analysis: {analysis_name}"

        help_text = build_help_text(prologue=help_text, args=[], model_help=False)

        # Build the actual function that will be the command
        run_analysis_command = build_analysis_command(
            analysis_binding=analysis_binding,
            help_text=help_text,
            pipeline=pipeline,
        )

        # Add the `--configuration` option for the analysis
        config = getattr(
            analysis_binding.analysis,
            "configuration_help",
            f'Path to a file with the configuration for the analysis "{analysis_name}".',
        )
        run_analysis_command = click.option(
            "-c",
            "--configuration",
            type=click.Path(exists=True, dir_okay=False, readable=True),
            default=None,
            callback=read_configuration_file,
            help=normalize_whitespace(config),
        )(run_analysis_command)

        # For each argument, call the `click.argument` decorator to dynamically add
        # them to the command
        for container_decl in analysis_binding.bindings:
            run_analysis_command = build_arg_objects(container_decl)(run_analysis_command)

        return run_analysis_command

    def _build_analysis_list_command(self, analysis_list_name: str, pipeline: Pipeline):
        """Dynamically create a command for running an analysis list."""
        analysis_list = pipeline.analysis_lists[analysis_list_name]

        if analysis_list.description:
            help_text = normalize_whitespace(analysis_list.description)
        else:
            help_text = f"Alias for [{', '.join(analysis_list.analyses)}]"

        help_text = build_help_text(prologue=help_text, args=[], model_help=False)

        # Compute all the container declarations required by all analyses in the list
        unique_container_decls: dict[str, ContainerDeclaration] = {}
        for analysis_name in analysis_list.analyses:
            analysis_binding = pipeline.analyses[analysis_name].bindings
            for container_decl in analysis_binding:
                unique_container_decls[container_decl.name] = container_decl

        # Build the actual function that will be the command
        run_analysis_command = build_analysis_list_command(
            analysis_list=analysis_list,
            container_decls=unique_container_decls,
            help_text=help_text,
            pipeline=pipeline,
        )

        return run_analysis_command


def build_analysis_list_command(
    analysis_list: AnalysisList,
    container_decls: dict[str, ContainerDeclaration],
    help_text: str,
    pipeline: Pipeline,
):
    analysis_name: str = analysis_list.name

    @click.command(
        cls=WrappablePypeCommand,
        name=analysis_name,
        help=help_text,
    )
    @output_option
    @invalidation_option
    @debug_option
    @list_objects_option
    @project_id_option
    @token_option
    @add_pipeline_config_options(pipeline, analysis_list)
    @exec_wrapper_if_needed
    @pass_context
    def run_analysis_command(
        ctx: ClickContext,
        project_id: str,
        token: str,
        runner_context: RunnerContext,
        output_file: IO[bytes],
        invalidations_file: IO[bytes] | None,
        **kwargs,
    ) -> None:
        pypeline_logger.debug_log(f'Running analysis: "{analysis_name}"')
        pypeline_logger.debug_log(f'and kwargs: "{kwargs}"')

        # Setup the backend factory (local compute or a remote daemon, per the
        # backend URL)
        backend_factory = backend_factory_for(
            ctx.obj.storage_provider_url,
            pipeline=ctx.obj.pipeline,
            base_directory=ctx.obj.base_directory,
            cache_dir=ctx.obj.cache_dir,
        )
        if runner_context.use_system and BackendFeature.DEBUG not in backend_factory.features:
            raise click.UsageError(
                "--debug is not supported by the selected backend: pipes and "
                "analyses do not run locally, so there is nothing to inspect."
            )
        if (
            invalidations_file is not None
            and BackendFeature.INVALIDATIONS not in backend_factory.features
        ):
            raise click.UsageError(
                "--invalidations is not supported by the selected backend: the "
                "daemon does not report which objects an analysis invalidated."
            )

        asyncio.run(
            async_part_of_command(
                backend_factory=backend_factory,
                project_id=project_id,
                token=token,
                runner_context=runner_context,
                pipeline=pipeline,
                analysis=analysis_list,
                configuration=ctx.obj.configuration,
                container_decls=tuple(container_decls.values()),
                output_file=output_file,
                invalidations_file=invalidations_file,
                kwargs=kwargs,
            )
        )

    return run_analysis_command


def build_analysis_command(
    analysis_binding: AnalysisBinding,
    help_text: str,
    pipeline: Pipeline,
):
    analysis_name: str = analysis_binding.analysis.name

    @click.command(
        cls=WrappablePypeCommand,
        name=analysis_name,
        help=help_text,
    )
    @output_option
    @invalidation_option
    @debug_option
    @list_objects_option
    @project_id_option
    @token_option
    @add_pipeline_config_options(pipeline, analysis_binding)
    @exec_wrapper_if_needed
    @pass_context
    def run_analysis_command(
        ctx: ClickContext,
        configuration: str,
        project_id: str,
        token: str,
        runner_context: RunnerContext,
        output_file: IO[bytes],
        invalidations_file: IO[bytes] | None,
        **kwargs,
    ) -> None:
        pypeline_logger.debug_log(f'Running analysis: "{analysis_name}"')
        pypeline_logger.debug_log(f'configuration: "{configuration}"')
        pypeline_logger.debug_log(f'and kwargs: "{kwargs}"')

        # Patch configuration
        ctx.obj.configuration[analysis_binding.analysis] = configuration

        # Setup the backend factory (local compute or a remote daemon, per the
        # backend URL)
        backend_factory = backend_factory_for(
            ctx.obj.storage_provider_url,
            pipeline=ctx.obj.pipeline,
            base_directory=ctx.obj.base_directory,
            cache_dir=ctx.obj.cache_dir,
        )
        if runner_context.use_system and BackendFeature.DEBUG not in backend_factory.features:
            raise click.UsageError(
                "--debug is not supported by the selected backend: pipes and "
                "analyses do not run locally, so there is nothing to inspect."
            )
        if (
            invalidations_file is not None
            and BackendFeature.INVALIDATIONS not in backend_factory.features
        ):
            raise click.UsageError(
                "--invalidations is not supported by the selected backend: the "
                "daemon does not report which objects an analysis invalidated."
            )
        asyncio.run(
            async_part_of_command(
                backend_factory=backend_factory,
                project_id=project_id,
                token=token,
                runner_context=runner_context,
                pipeline=pipeline,
                analysis=analysis_binding.analysis.name,
                container_decls=analysis_binding.bindings,
                configuration=ctx.obj.configuration,
                output_file=output_file,
                invalidations_file=invalidations_file,
                kwargs=kwargs,
            )
        )

    return run_analysis_command


@click.group(
    cls=AnalyzeGroup,
    help="Run an analysis",
)
def analyze() -> None:
    pass
