#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import asyncio
import sys
from pathlib import Path
from typing import AsyncContextManager

import click

from revng.pypeline.cli.common_options import add_pipeline_config_options
from revng.pypeline.cli.common_options import container_format_options, debug_option, full_help
from revng.pypeline.cli.common_options import list_objects_option, project_id_option, token_option
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.utils import build_help_text, detect_autocomplete, normalize_whitespace
from revng.pypeline.cli.wrappers import WrappablePypeCommand, exec_wrapper_if_needed
from revng.pypeline.container import ContainerFormat
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.object import ObjectID, ObjectSet
from revng.pypeline.pipeline import Artifact, Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.registry import get_singleton


class ArtifactGroup(click.Group):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx: ClickContext):  # type: ignore
        base = super().list_commands(ctx)
        pipeline = ctx.obj.pipeline
        if ctx.obj.show_hidden or detect_autocomplete(ctx):
            return base + sorted(pipeline.artifacts.keys())
        else:
            result = [*base]
            for artifact_name, artifact in pipeline.artifacts.items():
                if artifact.category.show_by_default:
                    result.append(artifact_name)
            return sorted(result)

    def get_command(self, ctx, cmd_name):  # type: ignore
        pipeline = ctx.obj.pipeline
        if cmd_name not in pipeline.artifacts:
            return super().get_command(ctx, cmd_name)
        return self._build_artifact_command(
            artifact_name=cmd_name,
            pipeline=pipeline,
        )

    def _build_artifact_command(self, artifact_name: str, pipeline: Pipeline):
        """Dynamically create a command for getting an artifact."""
        artifact: Artifact = pipeline.artifacts[artifact_name]

        if artifact.description is not None:
            help_text = click.wrap_text(f"\n{normalize_whitespace(artifact.description)}")
        else:
            help_text = f"Get the artifact: {artifact_name}"

        help_text = build_help_text(
            prologue=help_text,
            args=[],
            extra_args=["OBJECTS: Comma-separated list of object IDs to produce (default: all)"],
            model_help=False,
        )

        # Build the actual function that will be the command
        run_artifact_command = build_artifact_command(
            artifact=artifact,
            help_text=help_text,
            pipeline=pipeline,
        )

        # Add the `objects` argument to the command to specify the objects to produce
        run_artifact_command = click.argument(
            "objects",
            type=str,
            default=None,
            required=False,
        )(run_artifact_command)

        return run_artifact_command


def build_artifact_command(
    artifact: Artifact,
    help_text: str,
    pipeline: Pipeline,
):
    artifact_name: str = artifact.name

    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
        objects: str | None,
        configuration: PipelineConfiguration,
        result_path: Path | None,
        container_format: ContainerFormat,
        runner_context: RunnerContext,
        kwargs,
    ):
        """Since the storage provider factory returns an async context manager,
        we need the code that uses the storage_provider to be an async function.
        """
        async with storage_provider_context as storage_provider:
            loaded_model = storage_provider.get_model()[0]
            pypeline_logger.debug_log(f'Model loaded: "{loaded_model}"')

            artifact_kind = artifact.container.container_type.kind
            if kwargs["list"]:
                # If the user requested to list the available objects, we print them
                # and exit
                print(f'Available objects for kind: "{artifact_kind.__name__}"')
                for obj in loaded_model.all_objects(artifact_kind):
                    print(f" - {obj}")
                return

            # Compute the requests for the incoming containers of the
            # analysis
            incoming: ObjectSet

            if objects is None:
                incoming = loaded_model.all_objects(artifact_kind)
            else:
                obj_id_type = get_singleton(ObjectID)  # type: ignore[type-abstract]
                incoming = ObjectSet(
                    kind=artifact_kind,
                    objects={
                        obj_id_type.deserialize(obj)
                        for obj in objects.split(",")
                        if obj.strip() != ""
                    },
                )

            # Finally, run the analysis
            res_container = pipeline.get_artifact(
                model=ReadOnlyModel(loaded_model),
                artifact=artifact,
                requests=incoming,
                configuration=configuration,
                storage_provider=storage_provider,
                runner_context=runner_context,
            )
            pypeline_logger.debug_log("Artifact computed")

            if result_path is not None:
                pypeline_logger.debug_log(f'Writing result to: "{result_path}"')
                res_container.to_file(result_path, container_format=container_format)
            else:
                # Write to stdout the bytes of the container
                sys.stdout.buffer.write(res_container.to_bytes(container_format=container_format))
                sys.stdout.buffer.flush()

    @click.command(
        cls=WrappablePypeCommand,
        name=artifact_name,
        help=help_text,
        context_settings={
            "show_default": True,
        },
    )
    @list_objects_option
    @project_id_option
    @token_option
    @click.option(
        "-o",
        "result_path",
        type=click.Path(dir_okay=False, writable=True),
        help=(
            "Path to write the computed artifacts to, if not specified, the "
            "result will be printed to stdout. "
            "The default container_format when printing to stdout is json."
        ),
    )
    @debug_option
    @container_format_options
    @add_pipeline_config_options(pipeline, artifact.node)
    @exec_wrapper_if_needed
    @pass_context
    def run_artifact_command(
        ctx: ClickContext,
        project_id: str,
        token: str,
        objects: str | None,
        result_path: Path | None,
        container_format: ContainerFormat,
        runner_context: RunnerContext,
        **kwargs,
    ) -> None:
        pypeline_logger.debug_log(f'Running artifact: "{artifact_name}"')
        pypeline_logger.debug_log(f'container_format: "{container_format}"')
        pypeline_logger.debug_log(f'kwargs: "{kwargs}"')

        # Setup the storage provider
        storage_provider_factory = storage_provider_factory_factory(ctx.obj.storage_provider_url)
        storage_provider_context = storage_provider_factory.get(
            base_directory=ctx.obj.base_directory,
            project_id=project_id,
            token=token,
            cache_dir=ctx.obj.cache_dir,
        )
        # Switch to the async portion
        asyncio.run(
            async_part_of_command(
                storage_provider_context=storage_provider_context,
                objects=objects,
                configuration=ctx.obj.configuration,
                result_path=result_path,
                container_format=container_format,
                runner_context=runner_context,
                kwargs=kwargs,
            )
        )

    return run_artifact_command


@click.group(
    cls=ArtifactGroup,
    help="Compute an Artifact",
)
@full_help
def artifact() -> None:
    pass
