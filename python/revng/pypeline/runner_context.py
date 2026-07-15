#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import shlex
import subprocess
import tarfile
from collections.abc import Buffer
from pathlib import Path
from typing import Callable

import yaml

from revng.pypeline.analysis import Analysis
from revng.pypeline.container import Configuration, Container, ContainerFormat
from revng.pypeline.model import Model, ModelPath, ReadOnlyModel
from revng.pypeline.object import ObjectID, ObjectSet
from revng.pypeline.storage.file_provider import FileProvider
from revng.pypeline.task.pipe import Pipe, PipeDependencies
from revng.pypeline.task.task import TaskArgumentAccess
from revng.pypeline.utils import PypelineException
from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.naming import normalize_flag
from revng.pypeline.utils.registry import get_singleton

# Function defined by the pipebox that transforms the command-line before
# `run-pipe` and `run-analysis` are invoked. This receives the `argv` of the
# soon-to-be executed command and returns the modified argv.
ArgvHook = Callable[[list[str]], list[str]]


# Context class that is used for running pipe and analyses.
# When constructed with the default argument, and specifically
# `use_system=False`, the context will forward the arguments directly to the
# underlying pipe/analysis in the respective `run_pipe`/`run_analysis`.
# When `use_system=True`, it does not call the pipes directly, rahter it
# serializes all the container's contents and invokes the `run-pipe` command;
# this allows debugging individual pipe/analysis failures amidst a possibly
# long schedule.
class RunnerContext:
    def __init__(
        self,
        use_system: bool = False,
        argv_hook: ArgvHook | None = None,
        base_directory: Path | None = None,
    ):
        self.use_system = use_system
        if use_system:
            assert base_directory is not None
            assert base_directory.is_dir() or not base_directory.exists()
            base_directory.mkdir(parents=True, exist_ok=True)

            self.base_directory = base_directory
            self.argv_hook = argv_hook

        self.counter = 0

    def _get_directory(self, name: str) -> Path:
        assert self.use_system

        result = self.base_directory / f"{self.counter:04}-{name}"
        result.mkdir()
        self.counter += 1
        return result

    def _run_command(self, root_directory: Path, cmd: list[str]):
        assert self.use_system

        # Apply argv_hook if present
        if self.argv_hook is not None:
            cmd = self.argv_hook(cmd)

        # Store the reproducer script
        script_contents = """#!/usr/bin/env bash
set -euo pipefail

cd "$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

exec \\
"""
        for part in cmd:
            script_contents += f"  {shlex.quote(part)} \\\n"
        script_contents += '  "$@"\n'

        script_path = root_directory / "run"
        assert not script_path.is_file()
        script_path.write_text(script_contents)
        script_path.chmod(0o755)

        pypeline_logger.debug_log(f"Running debug command: {cmd}")
        result = subprocess.run(cmd, cwd=root_directory)
        if result.returncode != 0:
            pypeline_logger.log("\nDebug command failed! Re-run it with:")
            pypeline_logger.log(str(script_path.resolve()))
            raise PypelineException("Command execution failed")

    def _pipe_prepare_debug_directory(
        self,
        directory: Path,
        pipe: Pipe,
        model: ReadOnlyModel,
        file_provider: FileProvider,
        containers: list[Container],
    ):
        model_type = get_singleton(Model)  # type: ignore[type-abstract]

        # Populate the temporary directory with the files needed by the
        # pipe. The structure is:
        # files/          - subdirectory for --file-storage
        #   XXXXXXXXXXXXX - hashed file #1
        #   XXXXXXXXXXXXX - hashed file #2
        # inputs/         - subdirectory, input containers
        #   foo.tar       - input foo container, if `access & READ`
        # outputs/        - subdirectory, output containers
        #   foo.tar       - output foo container, if `access & WRITE`
        # <model name>    - model file

        # Store the files needed by the pipe
        file_requests = pipe.needed_files(model)
        files = file_provider.get_files(file_requests)
        files_dir = directory / "files"
        files_dir.mkdir()
        for file_request in file_requests:
            hash_ = file_request.hash
            (files_dir / hash_).write_bytes(files[hash_])

        # Serialize the model
        model_path = directory / model_type.model_name()
        model_path.write_bytes(model.serialize())

        # Create an `inputs` and `outputs` directory
        (directory / "inputs").mkdir()
        (directory / "outputs").mkdir()

        # For each argument if it needs to be read serialize the container
        # to disk
        for index, argument in enumerate(pipe.signature()):
            arg_name = normalize_flag(argument.name)
            if argument.access & TaskArgumentAccess.READ:
                output_path = directory / f"inputs/{arg_name}.tar"
                containers[index].to_file(output_path, container_format=ContainerFormat.TAR)

    def _pipe_construct_cmd(
        self,
        directory: Path,
        pipe: Pipe,
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> list[str]:
        cmd = ["pype", "pipeline", "run-pipe", pipe.name, "--tar"]
        # `run-pipe` reads its configuration from files: write them next to the
        # other inputs and pass their paths, relative to the working directory.
        if pipe.static_configuration != "":
            (directory / "static-configuration.yml").write_text(pipe.static_configuration)
            cmd += ["--static-configuration", "static-configuration.yml"]
        if configuration != "":
            (directory / "configuration.yml").write_text(configuration)
            cmd += ["--configuration", "configuration.yml"]
        cmd += ["--dependencies", "dependencies.tar"]
        cmd += ["--file-storage", "files"]

        model_type = get_singleton(Model)  # type: ignore[type-abstract]
        cmd_epilogue = [model_type.model_name()]

        for index, argument in enumerate(pipe.signature()):
            arg_name = normalize_flag(argument.name)
            if argument.access & TaskArgumentAccess.READ:
                cmd_epilogue.append(f"inputs/{arg_name}.tar")

            if argument.access & TaskArgumentAccess.WRITE:
                cmd_epilogue.append(f"outputs/{arg_name}.tar")
                argument_outgoing = outgoing[index]
                if len(argument_outgoing) > 0:
                    cmd += [
                        f"--{arg_name}-objects",
                        ",".join(x.serialize() for x in argument_outgoing),
                    ]

        return [*cmd, *cmd_epilogue]

    def _pipe_load_results(
        self, directory: Path, pipe: Pipe, containers: list[Container]
    ) -> PipeDependencies:
        object_id_type = get_singleton(ObjectID)  # type: ignore[type-abstract]
        # Load outputs back into the containers
        for index, argument in enumerate(pipe.signature()):
            if argument.access & TaskArgumentAccess.WRITE:
                arg_name = normalize_flag(argument.name)
                output_path = directory / f"outputs/{arg_name}.tar"
                containers[index] = containers[index].from_file(output_path, ContainerFormat.TAR)

        # Deserialize the dependency data produced by the pipe
        dependencies: list[list[tuple[ObjectID, ModelPath]]] = [[] for _ in pipe.signature()]
        custom_invalidation: list[list[tuple[ObjectID, Buffer]]] = [[] for _ in pipe.signature()]
        with tarfile.open(directory / "dependencies.tar", "r") as tar:
            for tar_info in tar:
                tar_member_file = tar.extractfile(tar_info)
                assert tar_member_file is not None

                if tar_info.name == "dependencies.yml":
                    dependency_data = yaml.safe_load(tar_member_file)
                    for index, raw_object_id, dependency_model_path in dependency_data:
                        object_id = object_id_type.deserialize(raw_object_id)
                        dependencies[index].append((object_id, dependency_model_path))
                else:
                    raw_index, raw_object_id = tar_info.name.split("/", 1)
                    custom_invalidation[int(raw_index)].append(
                        (
                            object_id_type.deserialize(f"/{raw_object_id}"),
                            tar_member_file.read(),
                        )
                    )

        return PipeDependencies(dependencies, custom_invalidation)

    def run_pipe(
        self,
        pipe: Pipe,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeDependencies:
        if self.use_system:
            temp_dir_path = self._get_directory(pipe.name)

            self._pipe_prepare_debug_directory(
                temp_dir_path, pipe, model, file_provider, containers
            )
            cmd = self._pipe_construct_cmd(temp_dir_path, pipe, outgoing, configuration)

            self._run_command(temp_dir_path, cmd)

            return self._pipe_load_results(temp_dir_path, pipe, containers)
        else:
            model.enable_caching()
            return pipe.run(
                file_provider=file_provider,
                model=model,
                containers=containers,
                incoming=incoming,
                outgoing=outgoing,
                configuration=configuration,
            )

    def run_analysis(
        self,
        analysis: Analysis,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        configuration: str,
    ) -> Model:
        if self.use_system:
            temp_dir_path = self._get_directory(analysis.name)
            model_type = get_singleton(Model)  # type: ignore[type-abstract]
            model_name = model_type.model_name()

            cmd = ["pype", "pipeline", "run-analysis", analysis.name]
            if configuration != "":
                configuration_path = temp_dir_path / "configuration.yml"
                configuration_path.write_text(configuration)
                cmd += ["--configuration", str(configuration_path)]
            output_model_path = temp_dir_path / ("output_" + model_name)
            cmd += ["-o", str(output_model_path)]

            # Serialize the model
            model_path = temp_dir_path / model_name
            model_path.write_bytes(model.serialize())
            cmd_epilogue = [str(model_path)]

            # For each argument, serialize its contents and append the requests
            # to the CLI
            for index, argument in enumerate(analysis.signature()):
                arg_name = normalize_flag(argument.name)
                output_path = temp_dir_path / f"{arg_name}.tar"
                containers[index].to_file(output_path, container_format=ContainerFormat.TAR)
                cmd_epilogue.append(str(output_path))

                argument_requests = incoming[index]
                if len(argument_requests) > 0:
                    cmd += [
                        f"--{arg_name}-objects",
                        ",".join(x.serialize() for x in argument_requests),
                    ]

            self._run_command(temp_dir_path, [*cmd, *cmd_epilogue])

            with open(output_model_path, "rb") as f:
                new_model = model_type.deserialize(f.read())[0]
        else:
            # The analysis modifies the model, but we need to invalidate
            # the changes, so we make a clone of the model.
            # This also allows to keep the original model intact
            # in case the analysis fails
            new_model = model.clone()
            new_model.disable_caching()
            try:
                analysis.run(
                    model=new_model,
                    containers=containers,
                    incoming=incoming,
                    configuration=configuration,
                )
            # TODO: eventually the pipe will raise `PypelineException` directly
            except RuntimeError as e:
                raise PypelineException(f"Analysis {analysis.name} failed to run: {e}")

        return new_model
