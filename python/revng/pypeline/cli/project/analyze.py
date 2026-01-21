#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import sys
from typing import AsyncContextManager

import click

from revng.pypeline.cli.common_options import debug_option, list_objects_option, project_id_option
from revng.pypeline.cli.common_options import token_option
from revng.pypeline.cli.utils import PypeGroup, build_arg_objects, build_help_text
from revng.pypeline.cli.utils import compute_objects, list_objects_for_container, normalize_flag
from revng.pypeline.cli.utils import normalize_pos_arg_name, normalize_whitespace
from revng.pypeline.cli.wrappers import WrappablePypeCommand, exec_wrapper_if_needed
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.pipeline import AnalysisBinding, AnalysisList, ContainerDeclaration, Pipeline
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.registry import get_singleton


class AnalyzeGroup(PypeGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        pipeline = ctx.obj.get("pipeline")
        if pipeline is None:
            return base
        return base + sorted(pipeline.analysis_lists.keys()) + sorted(pipeline.analyses.keys())

    def get_command(self, ctx, cmd_name):
        pipeline = ctx.obj.get("pipeline")
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
            model_type=get_singleton(Model),  # type: ignore[type-abstract]
            pipeline=pipeline,
        )

        config = getattr(
            analysis_binding.analysis,
            "configuration_help",
            f'Configuration for the analysis "{analysis_name}".',
        )
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
            model_type=get_singleton(Model),  # type: ignore[type-abstract]
            pipeline=pipeline,
        )

        # TODO: allow per-analysis configuration
        for analysis_name in analysis_list.analyses:
            config = f'Configuration for the analysis list "{analysis_list_name}".'
            run_analysis_command = click.option(
                f"--{normalize_flag(analysis_name)}-configuration",
                f"{normalize_pos_arg_name(analysis_name)}_configuration",
                type=str,
                default="",
                help=normalize_whitespace(config),
            )(run_analysis_command)

        return run_analysis_command


def build_analysis_list_command(
    analysis_list: AnalysisList,
    container_decls: dict[str, ContainerDeclaration],
    help_text: str,
    model_type: type[Model],
    pipeline: Pipeline,
):
    analysis_name: str = analysis_list.name

    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
        runner_context: RunnerContext,
        kwargs,
    ):
        """Since the storage provider factory returns an async context manager,
        we need the code that uses the storage_provider to be an async function.
        """
        async with storage_provider_context as storage_provider:
            loaded_model = model_type.deserialize(storage_provider.get_model()[0])

            pypeline_logger.debug_log(f'Model loaded: "{loaded_model}"')

            if kwargs["list"]:
                # If the user requested to list the available objects, we print them
                # and exit
                for container_decl in container_decls.values():
                    list_objects_for_container(
                        model=ReadOnlyModel(loaded_model),
                        arg_name=container_decl.name,
                        kind=container_decl.container_type.kind,
                    )
                    # Space between containers
                    print()
                return

            analysis_configuration = [
                kwargs[f"{normalize_pos_arg_name(analysis_name)}_configuration"]
                for analysis_name in analysis_list.analyses
            ]

            # Finally, run the analysis
            new_model, invalidated = pipeline.run_analysis_list(
                model=ReadOnlyModel(loaded_model),
                analysis_list=analysis_list,
                analysis_configuration=analysis_configuration,
                pipeline_configuration={},
                storage_provider=storage_provider,
                runner_context=runner_context,
            )

            pypeline_logger.debug_log("Analysis run completed")
            # Print on stdout the raw bytes of the modified model
            sys.stdout.buffer.write(new_model.serialize())

            # TODO: how to output this in a machine readable way?
            for container_location, object_ids in invalidated.items():
                serialized_ids = (object_id.serialize() for object_id in object_ids)
                pypeline_logger.debug_log(
                    f"Invalidated {container_location}: [{', '.join(serialized_ids)}]"
                )

    @click.command(
        cls=WrappablePypeCommand,
        name=analysis_name,
        help=help_text,
    )
    @debug_option
    @list_objects_option
    @project_id_option
    @token_option
    @exec_wrapper_if_needed
    @click.pass_context
    def run_analysis_command(
        ctx: click.Context,
        project_id: str,
        token: str,
        runner_context: RunnerContext,
        **kwargs,
    ) -> None:
        pypeline_logger.debug_log(f'Running analysis: "{analysis_name}"')
        pypeline_logger.debug_log(f'and kwargs: "{kwargs}"')

        # Load the model
        storage_provider_factory = storage_provider_factory_factory(ctx.obj["storage_provider"])
        storage_provider_context = storage_provider_factory.get(
            project_id=project_id,
            token=token,
            cache_dir=ctx.obj["cache_dir"],
        )
        asyncio.run(
            async_part_of_command(
                storage_provider_context=storage_provider_context,
                runner_context=runner_context,
                kwargs=kwargs,
            )
        )

    return run_analysis_command


def build_analysis_command(
    analysis_binding: AnalysisBinding,
    help_text: str,
    model_type: type[Model],
    pipeline: Pipeline,
):
    analysis_name: str = analysis_binding.analysis.name

    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
        configuration: str,
        runner_context: RunnerContext,
        kwargs,
    ):
        """Since the storage provider factory returns an async context manager,
        we need the code that uses the storage_provider to be an async function.
        """
        async with storage_provider_context as storage_provider:
            loaded_model = model_type.deserialize(storage_provider.get_model()[0])

            pypeline_logger.debug_log(f'Model loaded: "{loaded_model}"')

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
            new_model, invalidated = pipeline.run_analysis(
                model=ReadOnlyModel(loaded_model),
                analysis_name=analysis_name,
                requests=incoming,
                analysis_configuration=configuration,
                pipeline_configuration={},
                storage_provider=storage_provider,
                runner_context=runner_context,
            )
            pypeline_logger.debug_log("Analysis run completed")
            # Print on stdout the raw bytes of the modified model
            sys.stdout.buffer.write(new_model.serialize())

            # TODO: how to output this in a machine readable way?
            for container_location, object_ids in invalidated.items():
                serialized_ids = (object_id.serialize() for object_id in object_ids)
                pypeline_logger.debug_log(
                    f"Invalidated {container_location}: [{', '.join(serialized_ids)}]"
                )

    @click.command(
        cls=WrappablePypeCommand,
        name=analysis_name,
        help=help_text,
    )
    @debug_option
    @list_objects_option
    @project_id_option
    @token_option
    @exec_wrapper_if_needed
    @click.pass_context
    def run_analysis_command(
        ctx: click.Context,
        configuration: str,
        project_id: str,
        token: str,
        runner_context: RunnerContext,
        **kwargs,
    ) -> None:
        pypeline_logger.debug_log(f'Running analysis: "{analysis_name}"')
        pypeline_logger.debug_log(f'configuration: "{configuration}"')
        pypeline_logger.debug_log(f'and kwargs: "{kwargs}"')

        # Load the model
        storage_provider_factory = storage_provider_factory_factory(ctx.obj["storage_provider"])
        storage_provider_context = storage_provider_factory.get(
            project_id=project_id,
            token=token,
            cache_dir=ctx.obj["cache_dir"],
        )
        asyncio.run(
            async_part_of_command(
                storage_provider_context=storage_provider_context,
                configuration=configuration,
                runner_context=runner_context,
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
