#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import os
import sys
from pathlib import Path
from typing import IO, AsyncContextManager

import click
import yaml

from revng.pypeline.analysis import Analysis
from revng.pypeline.cli.backend import BackendFeature, backend_factory_for
from revng.pypeline.cli.common_options import AllAnalysesOption, add_pipeline_config_options
from revng.pypeline.cli.common_options import container_format_options, debug_option, full_help
from revng.pypeline.cli.common_options import project_id_option, token_option
from revng.pypeline.cli.context import ClickContext, pass_context
from revng.pypeline.cli.project import parse_pipeline_path
from revng.pypeline.cli.project.artifact import ArtifactGroup as ProjectArtifactGroup
from revng.pypeline.cli.utils import EagerParsedPath, build_arg_objects, compute_objects
from revng.pypeline.cli.utils import load_pipebox, load_pipeline, normalize_flag
from revng.pypeline.cli.utils import sort_option_groups
from revng.pypeline.cli.wrappers import WrappablePypeCommand, exec_with_wrapper
from revng.pypeline.cli.wrappers import exec_wrapper_if_needed
from revng.pypeline.container import ContainerDeclaration, ContainerFormat
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.local_provider import LocalStorageProviderFactory
from revng.pypeline.storage.storage_provider import FileStorageEntry, LockType, StorageProvider
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.storage.util import compute_hash
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.task import TaskArgumentAccess
from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.registry import get_singleton
from revng.support import get_root


def generate_model_with_binaries(binaries: list[Path]):
    result = []
    for index, binary in enumerate(binaries):
        with open(binary, "rb") as f:
            hash_ = compute_hash(f)
            size = f.seek(0, os.SEEK_END)
        result.append(
            {"Index": index, "Hash": hash_, "Size": size, "CanonicalPath": str(binary.resolve())}
        )

    return {"Binaries": result}


@click.group(help="Quick commands (japanese toilet)")
@click.option(
    "--pipeline",
    "pipeline",
    type=EagerParsedPath(name="pipeline", parser=parse_pipeline_path),
    help='Path to the pipeline file. Defaults to the "PYPELINE_PIPELINE" environment if set',
    default=Path(__file__).parent.parent / "pipeline.yml",
    envvar="PYPELINE_PIPELINE",
    show_default=True,
    expose_value=False,
)
@pass_context
def quick(ctx: ClickContext):
    load_pipebox(ctx)
    load_pipeline(ctx)


def handle_analysis_argument(
    pipeline: Pipeline,
    analyses: str | None,
    binary: Path,
    configuration: PipelineConfiguration,
    storage_provider: StorageProvider,
    runner_context: RunnerContext,
) -> Model:
    model_type = get_singleton(Model)  # type: ignore[type-abstract]
    model_raw = yaml.safe_dump(generate_model_with_binaries([binary])).encode()
    model = model_type.deserialize(model_raw)[0]
    if analyses is not None:
        if analyses == "":
            return model

        analyses_list = analyses.split(",")
        for analysis_name in analyses_list:
            if analysis_name not in pipeline.analyses:
                raise click.UsageError(f"Analysis {analysis_name} does not exist")

        for analysis_name in analyses_list:
            incoming = Requests()
            for container_decl in pipeline.analyses[analysis_name].bindings:
                incoming[container_decl] = compute_objects(
                    model=ReadOnlyModel(model),
                    arg_name=container_decl.name,
                    kind=container_decl.container_type.kind,
                    kwargs={},
                )

            model, _ = pipeline.run_analysis(
                model=ReadOnlyModel(model),
                analysis_name=analysis_name,
                requests=incoming,
                configuration=configuration,
                storage_provider=storage_provider,
                runner_context=runner_context,
            )
    else:
        analysis_list = pipeline.analysis_lists["initial-auto-analysis"]
        model, _ = pipeline.run_analysis_list(
            model=ReadOnlyModel(model),
            analysis_list=analysis_list,
            configuration=configuration,
            storage_provider=storage_provider,
            runner_context=runner_context,
        )

    return model


analyses_option = click.option(
    "--analyses",
    help=(
        "Instead of running the default initial auto analysis, run the "
        "specified list of (comma-separated) analyses"
    ),
)


def _build_artifact_command(pipeline: Pipeline, artifact_name: str):
    @quick.command(
        cls=WrappablePypeCommand,
        name=artifact_name,
        context_settings={
            "show_default": True,
        },
    )
    @click.argument("binary", type=click.Path(exists=True, dir_okay=False, path_type=Path))
    @click.option(
        "-o",
        "result_path",
        type=click.Path(dir_okay=False, writable=True),
        help=(
            "Path to write the computed artifacts to, if not specified, the "
            "result will be printed to stdout. "
            "The default output format is 'auto' (see --format)."
        ),
    )
    @debug_option
    @analyses_option
    @container_format_options(ContainerFormat.AUTO)
    @add_pipeline_config_options(
        pipeline, pipeline.artifacts[artifact_name].node, AllAnalysesOption.ALL_ANALYSES
    )
    @exec_wrapper_if_needed
    @pass_context
    def command(
        ctx: ClickContext,
        binary: Path,
        result_path: Path | None,
        analyses: str | None,
        container_format: ContainerFormat,
        runner_context: RunnerContext,
    ):
        pypeline_logger.debug_log(f'Running artifact: "{artifact_name}"')
        pypeline_logger.debug_log(f'container_format: "{container_format}"')

        async def async_part_of_command(
            storage_provider_context: AsyncContextManager[StorageProvider],
        ):
            async with storage_provider_context as storage_provider:
                # Add binaries to storage
                storage_provider.put_files_in_storage([FileStorageEntry(binary.name, binary)])

                # Run initial auto analysis
                new_model = handle_analysis_argument(
                    pipeline,
                    analyses,
                    binary,
                    ctx.obj.configuration,
                    storage_provider,
                    runner_context,
                )

                # Compute the requests and produce the artifacts
                artifact = pipeline.artifacts[artifact_name]
                artifact_kind = artifact.container.container_type.kind
                incoming: ObjectSet = new_model.all_objects(artifact_kind)
                res_container = pipeline.get_artifact(
                    model=ReadOnlyModel(new_model),
                    artifact=artifact,
                    requests=incoming,
                    configuration=ctx.obj.configuration,
                    storage_provider=storage_provider,
                    runner_context=runner_context,
                )
                pypeline_logger.debug_log("Artifact computed")

                if result_path is not None:
                    pypeline_logger.debug_log(f'Writing result to: "{result_path}"')
                    res_container.to_file(result_path, container_format=container_format)
                else:
                    # Write to stdout the bytes of the container
                    sys.stdout.buffer.write(
                        res_container.to_bytes(container_format=container_format)
                    )
                    sys.stdout.buffer.flush()

        storage_provider_factory = LocalStorageProviderFactory("local://?temporary")
        storage_provider_context = storage_provider_factory.get(
            ctx.obj.base_directory, ctx.obj.pipeline, LockType.ARTIFACT, None, None, None
        )
        asyncio.run(async_part_of_command(storage_provider_context))

    return command


class ArtifactGroup(ProjectArtifactGroup):
    def get_command(self, ctx: ClickContext, cmd_name):  # type: ignore
        pipeline = ctx.obj.pipeline
        if cmd_name in pipeline.artifacts:
            return _build_artifact_command(pipeline, cmd_name)
        else:
            return super().get_command(ctx, cmd_name)


@quick.group(cls=ArtifactGroup, help="Run analyses and compute an artifact")
@full_help
def artifact():
    pass


class AnalyzeCommand(WrappablePypeCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_options_added = False

    def get_params(self, ctx: ClickContext):  # type: ignore[override]
        if not self._config_options_added:
            add_pipeline_config_options(ctx.obj.pipeline, AllAnalysesOption.ALL_ANALYSES)(self)
            sort_option_groups(self)
            self._config_options_added = True
        return super().get_params(ctx)


@quick.command(
    cls=AnalyzeCommand,
    name="analyze",
    context_settings={
        "show_default": True,
    },
)
@click.argument("binary", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "output_file",
    type=click.File("wb"),
    help="Path to write the model to, if not specified, the result will be printed to stdout.",
    default="-",
)
@analyses_option
@debug_option
@exec_wrapper_if_needed
@pass_context
def analyze(
    ctx: ClickContext,
    binary: Path,
    output_file: IO[bytes],
    analyses: str | None,
    runner_context: RunnerContext,
):
    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
    ):
        async with storage_provider_context as storage_provider:
            # Add binaries to storage
            storage_provider.put_files_in_storage([FileStorageEntry(binary.name, binary)])

            # Run initial auto analysis
            model = handle_analysis_argument(
                ctx.obj.pipeline,
                analyses,
                binary,
                ctx.obj.configuration,
                storage_provider,
                runner_context,
            )
            output_file.write(model.serialize())

    storage_provider_factory = LocalStorageProviderFactory("local://?temporary")
    storage_provider_context = storage_provider_factory.get(
        ctx.obj.base_directory, ctx.obj.pipeline, LockType.ANALYSIS, None, None, None
    )
    asyncio.run(async_part_of_command(storage_provider_context))


@click.command(
    cls=WrappablePypeCommand,
    name="init",
)
@click.argument(
    "binary",
    required=False,
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--no-initial-auto-analysis", is_flag=True)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite an existing project model instead of failing.",
)
@project_id_option
@token_option
@exec_wrapper_if_needed
@pass_context
def init(
    ctx: ClickContext,
    binary: Path | None,
    no_initial_auto_analysis: bool,
    overwrite: bool,
    project_id: str,
    token: str,
):
    """Initialize a new project."""
    backend_factory = backend_factory_for(
        ctx.obj.storage_provider_url,
        pipeline=ctx.obj.pipeline,
        base_directory=ctx.obj.base_directory,
        cache_dir=ctx.obj.cache_dir,
    )
    if BackendFeature.INIT not in backend_factory.features:
        raise click.UsageError(
            "`init` cannot be used against a `daemon://` URL: a daemon manages "
            "its own project. Import the binary with an analysis instead."
        )

    model_type = get_singleton(Model)  # type: ignore[type-abstract]

    storage_provider_factory = storage_provider_factory_factory(ctx.obj.storage_provider_url)
    if not storage_provider_factory.init(ctx.obj.base_directory, overwrite):
        model_name = model_type.model_name()
        raise click.UsageError(
            f"File {model_name} is already present in the current directory. "
            "Refusing to overwrite it (pass --overwrite to replace it)."
        )

    async def async_part_of_command(
        storage_provider_context: AsyncContextManager[StorageProvider],
    ):
        async with storage_provider_context as storage_provider:
            if binary is not None:
                model_raw = yaml.safe_dump(generate_model_with_binaries([binary])).encode()
                model = model_type.deserialize(model_raw)[0]
                storage_provider.set_model(model, set(), [])

                file_entry = FileStorageEntry(binary.name, path=binary.resolve())
                storage_provider.put_files_in_storage([file_entry])

            if no_initial_auto_analysis:
                return

            pipeline = ctx.obj.pipeline
            analysis_list = pipeline.analysis_lists["initial-auto-analysis"]
            pipeline.run_analysis_list(
                model=ReadOnlyModel(model),
                analysis_list=analysis_list,
                configuration=ctx.obj.configuration,
                storage_provider=storage_provider,
            )

    storage_provider_context = storage_provider_factory.get(
        base_directory=ctx.obj.base_directory,
        pipeline=ctx.obj.pipeline,
        lock_type=LockType.ANALYSIS,
        project_id=project_id,
        token=token,
        cache_dir=ctx.obj.cache_dir,
    )
    asyncio.run(async_part_of_command(storage_provider_context))


def _generate_load_arguments(ctx):
    native_libraries: list[Path] = ctx.obj.pipebox._native_libraries
    return [f"-load={p.resolve()!s}" for p in native_libraries]


class RunPipeNativeGroup(click.Group):
    """
    This implements the "run-pipe-native" command group, subcommands of this
    group are exclusively native pipes that can be run without python.
    This command sets up the arguments and calls the
    libexec/pypeline-run-pipe executable (with wrappers if present).
    """

    @staticmethod
    def _get_pipes(ctx) -> dict[str, type[Pipe]]:
        return ctx.obj.pipebox._native_pipes

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        return base + sorted(self._get_pipes(ctx).keys())

    def get_command(self, ctx, cmd_name):
        if cmd_name in self._get_pipes(ctx):
            return self._build_pipe_command(ctx, cmd_name)
        return super().get_command(ctx, cmd_name)

    def _build_pipe_command(self, ctx, pipe_name: str):
        """Dynamically create a command for running a pipe."""
        pipe_type: type[Pipe] = self._get_pipes(ctx)[pipe_name]

        @click.command(
            cls=WrappablePypeCommand,
            name=pipe_name,
            help=f"Run the {pipe_name} pipe natively",
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        )
        @click.option("--tar", is_flag=True, expose_value=False)
        @click.pass_context
        def run_pipe_command(ctx, **kwargs):
            args = [
                str(get_root() / "libexec/revng/pypeline-run-pipe"),
                *_generate_load_arguments(ctx),
                pipe_name,
            ]
            for arg in pipe_type.signature():
                if arg.access == TaskArgumentAccess.READ:
                    continue

                normalized_name = normalize_flag(arg.name)
                objects_arg = f"{normalized_name}_objects"
                if objects_arg in kwargs and kwargs[objects_arg] is not None:
                    args.extend(["--objects", normalized_name, kwargs[objects_arg]])
            args.extend(ctx.args)
            exec_with_wrapper(args)

        for arg in pipe_type.signature():
            if TaskArgumentAccess.WRITE in arg.access:
                run_pipe_command = build_arg_objects(arg)(run_pipe_command)

        return run_pipe_command


@click.group(
    cls=RunPipeNativeGroup,
    help="Run a pipe without python",
)
def run_pipe_native() -> None:
    pass


class RunAnalysisNativeGroup(click.Group):
    """
    This implements the "run-analysis-native" command group, subcommands of
    this group are exclusively native analyses that can be run without python.
    This command sets up the arguments and calls the
    libexec/pypeline-run-analysis executable (with wrappers if present).
    """

    @staticmethod
    def _get_analyses(ctx) -> dict[str, type[Analysis]]:
        return ctx.obj.pipebox._native_analyses

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        return base + sorted(self._get_analyses(ctx).keys())

    def get_command(self, ctx, cmd_name):
        if cmd_name in self._get_analyses(ctx):
            return self._build_analysis_command(ctx, cmd_name)
        return super().get_command(ctx, cmd_name)

    def _build_analysis_command(self, ctx, analysis_name: str):
        """Dynamically create a command for running a pipe."""
        analysis_type: type[Analysis] = self._get_analyses(ctx)[analysis_name]

        @click.command(
            cls=WrappablePypeCommand,
            name=analysis_name,
            help=f"Run the {analysis_name} analysis natively",
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        )
        @click.pass_context
        def run_analysis_command(ctx, **kwargs):
            args = [
                str(get_root() / "libexec/revng/pypeline-run-analysis"),
                *_generate_load_arguments(ctx),
                analysis_name,
            ]
            for container_type in analysis_type.signature():
                normalized_name = normalize_flag(container_type.name)
                objects_arg = f"{normalized_name}_objects"
                if objects_arg in kwargs and kwargs[objects_arg] is not None:
                    args.extend(["--objects", normalized_name, kwargs[objects_arg]])
            args.extend(ctx.args)
            exec_with_wrapper(args)

        for container_type in analysis_type.signature():
            arg = ContainerDeclaration(container_type.name, container_type)
            run_analysis_command = build_arg_objects(arg)(run_analysis_command)

        return run_analysis_command


@click.group(
    cls=RunAnalysisNativeGroup,
    help="Run an analysis without python",
)
def run_analysis_native() -> None:
    pass
