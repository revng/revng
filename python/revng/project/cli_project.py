#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import hashlib
import json
import re
import shutil
from contextlib import ExitStack
from pathlib import Path
from subprocess import PIPE
from tempfile import NamedTemporaryFile, TemporaryFile
from typing import Dict, List, Optional, Set, Union

import yaml

from revng.project.common import ALL_OBJECTS, AllObjects, CLIHelper
from revng.project.model import Binary  # type: ignore[attr-defined]
from revng.project.pipeline_description import PipelineDescription
from revng.project.project import Project
from revng.support import TarDictionary


class CLIProject(Project):
    """
    This class is used to run revng analysis and artifact through
    the revng CLI with subprocess.
    """

    def __init__(
        self, project_directory: Optional[str] = None, executable_path: Optional[str] = None
    ):
        self._cli_helper = CLIHelper(project_directory, executable_path)
        # The _cli_helper needs to be initialized first before calling the
        # parent's init since that calls `_get_pipeline_description`
        super().__init__()
        self._load_model()

    def upload_binary(self, binary_path: Union[str, Path]) -> str:
        binary_path = Path(binary_path)
        shutil.copy(binary_path, self._cli_helper.project_directory / binary_path.name)
        with open(binary_path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    def _get_artifact_impl(
        self,
        artifact_name: str,
        objects: Union[Set[str], AllObjects],
        configuration: Dict[str, str],
    ) -> Dict[str, bytes]:
        with NamedTemporaryFile() as tmp_file:
            args = ["artifact", artifact_name, "--tar", "-o", tmp_file.name]
            if objects is not ALL_OBJECTS:
                args.append(",".join(objects))
            if len(configuration) > 0:
                args.extend(("-c", json.dumps(configuration)))

            self._cli_helper.run(args)
            tmp_file.seek(0)
            return dict(TarDictionary(tmp_file))

    def _analyze_impl(
        self,
        analysis_name: str,
        configuration: Dict[str, str],
        containers: Dict[str, List[str]],
    ) -> int:
        args = ["analyze", analysis_name]

        # `-c` takes the path to a file with the configuration, so write it to a
        # temporary file and pass its path. The files are kept open until the
        # command has run.
        with ExitStack() as config_scope:
            for name, value in configuration.items():
                if name == analysis_name and analysis_name in self._analysis_names:
                    # Single-analysis specialization
                    config_file = config_scope.enter_context(NamedTemporaryFile("w", suffix=".yml"))
                    config_file.write(value)
                    config_file.flush()
                    args.extend(("-c", config_file.name))
                else:
                    args.extend((f"--{self._normalize_argument(value)}-configuration", value))

            for container_name, obj_list in containers.items():
                if len(obj_list) > 0:
                    args.extend(
                        (
                            f"--{self._normalize_argument(container_name)}-objects",
                            ",".join(obj_list),
                        )
                    )

            with TemporaryFile("w+") as temp_file:
                self._cli_helper.run(args, stdout=temp_file)
                temp_file.seek(0)
                model = Binary.deserialize(temp_file, self)

        self._set_model(model)
        return self._epoch + 1

    def _load_model(self):
        with open(self._cli_helper.model_path) as f:
            data = f.read()

        if data != "":
            model = Binary.deserialize(data, self)
            self._set_model(model)

    def _get_pipeline_description(self) -> PipelineDescription:
        process = self._cli_helper.run(["dump-pipeline"], stdout=PIPE)
        return yaml.safe_load(process.stdout)

    def _get_epoch(self) -> int:
        return 0

    @staticmethod
    def _normalize_argument(input_: str) -> str:
        return re.sub(r"\s+", " ", input_).strip().replace(" ", "-").replace("_", "-")
