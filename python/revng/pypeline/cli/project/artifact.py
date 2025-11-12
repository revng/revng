#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import asyncio
import sys
from pathlib import Path
from typing import AsyncContextManager

import click

from revng.pypeline.cli.utils import PypeGroup, build_help_text, container_format_option
from revng.pypeline.cli.utils import is_json_option, is_tar_option, list_objects_option
from revng.pypeline.cli.utils import normalize_whitespace, project_id_option, token_option
from revng.pypeline.container import ContainerFormat
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectID, ObjectSet
from revng.pypeline.pipeline import Artifact, Pipeline
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.registry import get_singleton


class ArtifactGroup(PypeGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        pipeline = ctx.obj.get("pipeline")
        if pipeline is None:
            return base
        return base + sorted(pipeline.artifacts.keys())

    def get_command(self, ctx, cmd_name):
        pipeline = ctx.obj.get("pipeline")
        if pipeline is None:
            return super().get_command(ctx, cmd_name)
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
            model_type=get_singleton(Model),  # type: ignore[type-abstract]
            pipeline=pipeline,
        )

        config = getattr(
            artifact,
            "configuration_help",
            f'Configuration for the artifact "{artifact_name}".',
        )
        if config is not None:
            run_artifact_command = click.option(
                "-c",
                "--configuration",
                type=str,
                default="",
                help=normalize_whitespace(config),
            )(run_artifact_command)

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
    model_type: type[Model],
    pipeline: Pipeline,
):
    artifact_name: str = artifact.name

    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
        objects: str | None,
        result_path: Path | None,
        container_format: ContainerFormat,
        kwargs,
    ):
        """Since the storage provider factory returns an async context manager,
        we need the code that uses the storage_provider to be an async function.
        """
        async with storage_provider_context as storage_provider:
            loaded_model = model_type.deserialize(storage_provider.get_model()[0])

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
                pipeline_configuration={},
                storage_provider=storage_provider,
            )
            pypeline_logger.debug_log("Artifact computed")

            if result_path is not None:
                pypeline_logger.debug_log(f'Writing result to: "{result_path}"')
                res_container.to_file(result_path, container_format)
            else:
                # While "auto" works great when writing to a file, when writing to stdout we cannot
                # infer the format, so we need to take a decision.
                # We default to json as it's the easiest to read on the terminal.
                if container_format == "auto":
                    container_format = "json"
                # Write to stdout the bytes of the container
                sys.stdout.buffer.write(res_container.to_bytes(container_format=container_format))
                sys.stdout.buffer.flush()

    @click.command(
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
    @container_format_option
    @is_tar_option
    @is_json_option
    @click.pass_context
    def run_analysis_command(
        ctx: click.Context,
        configuration: str,
        project_id: str,
        token: str,
        objects: str | None,
        result_path: Path | None,
        container_format: ContainerFormat,
        is_tar: bool,
        is_json: bool,
        **kwargs,
    ) -> None:
        if is_tar:
            container_format = "tar"
        if is_json:
            container_format = "json"
        pypeline_logger.debug_log(f'Running artifact: "{artifact_name}"')
        pypeline_logger.debug_log(f'configuration: "{configuration}"')
        pypeline_logger.debug_log(f'container_format: "{container_format}"')
        pypeline_logger.debug_log(f'kwargs: "{kwargs}"')

        # Setup the storage provider
        storage_provider_factory = storage_provider_factory_factory(ctx.obj["storage_provider"])
        storage_provider_context = storage_provider_factory.get(
            project_id=project_id,
            token=token,
            cache_dir=ctx.obj["cache_dir"],
        )
        # Switch to the async portion
        asyncio.run(
            async_part_of_command(
                storage_provider_context=storage_provider_context,
                objects=objects,
                result_path=result_path,
                container_format=container_format,
                kwargs=kwargs,
            )
        )

    return run_analysis_command


@click.group(
    cls=ArtifactGroup,
    help="Compute an Artifact",
)
def artifact() -> None:
    pass
