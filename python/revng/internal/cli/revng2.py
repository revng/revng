#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
This is just a wrapper over `pype` that sets pipebox to the revng pipebox path.
The path is computed relatively to this file, so this should work regardless of
where revng is installed.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncContextManager

import click
import yaml
from click.shell_completion import CompletionItem

from revng.internal.support import cache_directory
from revng.pypeline.cli.project import project
from revng.pypeline.cli.utils import EagerParsedPath, PypeCommand, PypeGroup
from revng.pypeline.cli.utils import container_format_options, project_id_option, token_option
from revng.pypeline.container import ContainerFormat
from revng.pypeline.main import pype, run
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.pipeline import Artifact, Pipeline
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file
from revng.pypeline.storage.local_provider import TemporaryLocalStorageProviderFactory
from revng.pypeline.storage.storage_provider import FileStorageEntry, StorageProvider
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.storage.util import compute_hash
from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.registry import get_singleton


def generate_model_with_binaries(binaries: list[Path]):
    result = []
    for index, binary in enumerate(binaries):
        with open(binary, "rb") as f:
            hash_ = compute_hash(f)
            size = f.seek(0, os.SEEK_END)
        result.append({"Index": index, "Hash": hash_, "Size": size, "Name": binary.name})

    return {"Binaries": result}


@click.group(cls=PypeGroup, help="Quick commands (japanese toilet)")
@click.option(
    "--pipeline",
    "pipeline",
    type=EagerParsedPath(
        name="pipeline",
        parser=lambda path, _ctx: load_pipeline_yaml_file(path),
    ),
    help='Path to the pipeline file. Defaults to the "PIPELINE" environment if set',
    default=os.environ.get("PIPELINE", Path(__file__).parent.parent / "pipeline.yml"),
    show_default=True,
)
@click.pass_context
def quick(
    ctx: click.Context,
    pipeline: Path,
):
    if ctx.obj is None:
        ctx.obj = {}
    # Store the params so the subcommands can access them
    ctx.obj.update({"pipeline": pipeline})


class ArtifactArgument(click.Argument):
    class ArtifactChoice(click.ParamType):
        def convert(self, value, param, ctx):
            if isinstance(value, Artifact):
                return value

            artifacts = ctx.obj["pipeline"].artifacts
            if value in artifacts:
                return artifacts[value]
            else:
                self.fail(f"{value} is not one of " + ", ".join(sorted(artifacts)))

        def shell_complete(self, ctx, param, incomplete: str) -> list[CompletionItem]:
            artifacts = ctx.obj["pipeline"].artifacts
            matched = (a for a in artifacts if a.startswith(incomplete))
            return [CompletionItem(c) for c in sorted(matched)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, type=self.__class__.ArtifactChoice())

    def get_help_record(self, ctx: click.Context) -> tuple[str, str] | None:
        text = "One of the following:\n\n\b\n"
        for artifact in sorted(ctx.obj["pipeline"].artifacts):
            text += f"* {artifact}\n"
        return (self.make_metavar(ctx), text)


@quick.command(
    cls=PypeCommand,
    name="artifact",
    context_settings={
        "show_default": True,
    },
)
@click.argument("binary", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("artifact", cls=ArtifactArgument)
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
@container_format_options
@click.pass_context
def artifact(
    ctx,
    binary: Path,
    artifact: Artifact,
    result_path: Path | None,
    container_format: ContainerFormat,
):
    pypeline_logger.debug_log(f'Running artifact: "{artifact}"')
    pypeline_logger.debug_log(f'container_format: "{container_format}"')
    pipeline: Pipeline = ctx.obj["pipeline"]
    model_type = get_singleton(Model)  # type: ignore[type-abstract]

    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
    ):
        async with storage_provider_context as storage_provider:
            # Add binaries to storage
            storage_provider.put_files_in_storage([FileStorageEntry(binary.name, binary)])

            # Run initial auto analysis
            model_raw = yaml.safe_dump(generate_model_with_binaries([binary])).encode()
            model = model_type.deserialize(model_raw)
            analysis_list = pipeline.analysis_lists["initial-auto-analysis"]
            analysis_configuration = ["" for _ in analysis_list.analyses]
            new_model, _ = pipeline.run_analysis_list(
                model=ReadOnlyModel(model),
                analysis_list=analysis_list,
                analysis_configuration=analysis_configuration,
                pipeline_configuration={},
                storage_provider=storage_provider,
            )

            # Compute the requests and produce the artifacts
            artifact_kind = artifact.container.container_type.kind
            incoming: ObjectSet = new_model.all_objects(artifact_kind)
            res_container = pipeline.get_artifact(
                model=ReadOnlyModel(new_model),
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
                # Write to stdout the bytes of the container
                sys.stdout.buffer.write(res_container.to_bytes(container_format=container_format))
                sys.stdout.buffer.flush()

    storage_provider_factory = TemporaryLocalStorageProviderFactory("temporary://")
    storage_provider_context = storage_provider_factory.get(None, None, None)
    asyncio.run(async_part_of_command(storage_provider_context))


@click.command()
@click.argument(
    "binary",
    required=False,
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--no-initial-auto-analysis", is_flag=True)
@project_id_option
@token_option
@click.pass_context
def init(ctx, binary: Path | None, no_initial_auto_analysis: bool, project_id: str, token: str):
    """Initialize a new project."""
    model_type = get_singleton(Model)  # type: ignore[type-abstract]
    model_name = model_type.model_name()
    model_file = Path.cwd() / model_name
    if model_file.exists():
        raise click.UsageError(
            f"File {model_name} is already present in the current directory. "
            "Refusing to overwrite it."
        )
    model_file.touch()

    if binary is not None:
        model_raw = yaml.safe_dump(generate_model_with_binaries([binary])).encode()
        with open(model_file, "wb") as f:
            f.write(model_raw)
    else:
        model_raw = b""

    if no_initial_auto_analysis:
        return

    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
    ):
        pipeline = ctx.obj["pipeline"]
        async with storage_provider_context as storage_provider:
            model = model_type.deserialize(model_raw)
            analysis_list = pipeline.analysis_lists["initial-auto-analysis"]
            analysis_configuration = ["" for _ in analysis_list.analyses]
            pipeline.run_analysis_list(
                model=ReadOnlyModel(model),
                analysis_list=analysis_list,
                analysis_configuration=analysis_configuration,
                pipeline_configuration={},
                storage_provider=storage_provider,
            )

    storage_provider_factory = storage_provider_factory_factory(ctx.obj["storage_provider"])
    storage_provider_context = storage_provider_factory.get(
        project_id=project_id,
        token=token,
        cache_dir=ctx.obj["cache_dir"],
    )
    asyncio.run(async_part_of_command(storage_provider_context))


def patch_pype():
    """
    revng2 is based on `pype`, but we want to change some defaults to be revng specific,
    and we want to add some commands.
    """
    # Replace the name (needed for autocompletion and usage)
    pype.name = "revng2"
    pype.add_command(quick)
    # Replace the default for pipebox
    for param in pype.params:
        if param.name == "pipebox":
            param.default = os.environ.get("PIPEBOX", Path(__file__).parent.parent / "pipebox.py")

    # Add `init` to project subcommand
    project.add_command(init)
    # Change the default for pipeline
    for param in project.params:
        if param.name == "pipeline":
            param.default = os.environ.get(
                "PIPELINE", Path(__file__).parent.parent / "pipeline.yml"
            )
        elif param.name == "cache_dir":
            param.default = cache_directory()


def main():
    """Entry point for revng2."""
    patch_pype()
    run()


if __name__ == "__main__":
    main()
