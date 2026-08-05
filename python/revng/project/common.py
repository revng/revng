#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shutil
from enum import Enum, auto
from pathlib import Path
from subprocess import PIPE, CalledProcessError, Popen, run
from tempfile import TemporaryDirectory
from typing import Iterable, Optional, Union

import yaml

from revng.project.pipeline_description import PipelineDescription


# This enum and the constant below are all needed to have OK typing in the
# public project interface. Eventually PEP661 (sentinel value) can be used once
# it's merged and has widespread support.
class AllObjects(Enum):
    ALL_OBJECTS = auto()


ALL_OBJECTS = AllObjects.ALL_OBJECTS
"""Marker value indicating that all the objects of the specified artifact must
be produced"""


class ProjectError(Exception):
    pass


class CLIError(ProjectError):
    pass


class HTTPError(ProjectError):
    pass


class CLIHelper:
    """Mixin for classes that use the `revng` command-line tool"""

    def __init__(self, project_directory: Union[str, Path, None], executable_path: Optional[str]):
        # Handle the executable_path parameter, using `which` if it's missing
        if not executable_path:
            _executable_path = shutil.which("revng")
        else:
            _executable_path = executable_path
        assert _executable_path is not None
        assert os.access(_executable_path, os.F_OK | os.X_OK)
        self._executable_path = _executable_path

        # Handle the project_directory parameter, using a temporary directory
        # if not supplied
        if project_directory is None:
            self._tmp_dir = TemporaryDirectory()
            project_directory = self._tmp_dir.name
        if not os.path.exists(project_directory):
            os.makedirs(project_directory)
        assert os.path.isdir(project_directory)
        self._project_directory = str(project_directory)

        # We will be invoking `revng project` a bunch. This family of commands
        # needs the file `${model_name}` to be present in the project
        # directory. Since the project directory might not contain a model,
        # touch one if missing.
        process = self.run(["dump-pipeline"], stdout=PIPE)
        pipeline_description: PipelineDescription = yaml.safe_load(process.stdout)
        model_name: str = pipeline_description["model"]["name"]
        self._model_path = Path(project_directory) / model_name
        if not self._model_path.exists():
            self._model_path.touch()

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def project_directory(self) -> Path:
        return Path(self._project_directory)

    def _make_command_line(self, cmd_args: Iterable[str]) -> list[str]:
        return [self._executable_path, "-C", self._project_directory, "project", *cmd_args]

    def run(self, cmd_args: Iterable[str], *args, stdout=None):
        cmdline = self._make_command_line(cmd_args)
        try:
            return run(  # type: ignore[call-overload]
                cmdline, *args, stdout=stdout, stderr=PIPE, check=True
            )
        except CalledProcessError as e:
            error_message = f"Command {' '.join(cmdline)} exited with code {e.returncode}\n"
            error_message += "Stderr output:\n"
            error_message += e.stderr.decode("utf8")
            raise CLIError(error_message) from e

    def popen(self, cmd_args: Iterable[str], *args, stdout, stderr) -> Popen:
        return Popen(  # type: ignore
            self._make_command_line(cmd_args), *args, stdout=stdout, stderr=stderr
        )
