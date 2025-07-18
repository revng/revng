#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from tempfile import NamedTemporaryFile
from typing import Optional, Set

import yaml

from revng.model import Binary, DiffSet  # type: ignore[attr-defined]
from revng.pipeline_description import PipelineDescription  # type: ignore[attr-defined]
from revng.pipeline_description import YamlLoader  # type: ignore[attr-defined]

from .project import CLIProjectMixin, Project, ResumeProjectMixin


class CLIProject(Project, CLIProjectMixin, ResumeProjectMixin):
    """
    This class is used to run revng analysis and artifact through
    the CLI with subprocess.
    """

    def __init__(
        self, resume_path: Optional[str] = None, revng_executable_path: Optional[str] = None
    ):
        CLIProjectMixin.__init__(self, revng_executable_path)
        ResumeProjectMixin.__init__(self, resume_path)
        Project.__init__(self)
        self._input_binary_path: Optional[str] = None
        self._load_model()

    def set_binary_path(self, binary_path: str):
        assert os.path.isfile(binary_path)
        self._input_binary_path = binary_path

    def _get_artifact_impl(self, artifact_name: str, targets: Set[str]) -> bytes:
        assert self._input_binary_path is not None
        args = [
            "artifact",
            f"--resume={self._resume_path}",
            artifact_name,
            self._input_binary_path,
            *targets,
        ]

        return self._run_revng_cli(args, capture_output=True).stdout

    def _analyze(self, analysis_name: str, targets={}, options={}):
        assert analysis_name in self._analysis_names
        self._analyze_impl(analysis_name, targets, options)

    def _analyses_list(self, analysis_list_name: str):
        assert analysis_list_name in self._analyses_list_names
        return self._analyze_impl(analysis_list_name)

    def _analyze_impl(self, analysis_name: str, targets={}, options={}):
        assert self._input_binary_path is not None
        # TODO: add `targets` when `revng` allows it
        assert len(targets) == 0

        args = ["analyze", f"--resume={self._resume_path}", analysis_name]
        if options:
            for name, value in options.items():
                args.append(f"--{analysis_name}-{name}={value}")
        args.append(self._input_binary_path)

        result = self._run_revng_cli(args, capture_output=True).stdout
        model = Binary.deserialize(result.decode("utf-8"))
        self._set_model(model)

    def _commit(self):
        diff = DiffSet.make(self._last_saved_model, self.model)
        if len(diff.Changes) == 0:
            return

        with NamedTemporaryFile(mode="w") as tmp_diff_file:
            diff.serialize(tmp_diff_file)
            tmp_diff_file.flush()

            args = [
                "analyze",
                f"--resume={self._resume_path}",
                "apply-diff",
                "--apply-diff-global-name=model.yml",
                f"--apply-diff-diff-content-path={tmp_diff_file.name}",
                self._input_binary_path,
            ]

            model = self._run_revng_cli(args, capture_output=True).stdout
            self._last_saved_model = Binary.deserialize(model.decode("utf-8"))

    def _get_pipeline_description(self) -> PipelineDescription:
        pipeline_description_path = f"{self._resume_path}/pipeline-description.yml"
        if not os.path.isfile(pipeline_description_path):
            self._run_revng_cli(["init", f"--resume={self._resume_path}"])
        assert os.path.isfile(pipeline_description_path)
        with open(pipeline_description_path) as f:
            return yaml.load(f, Loader=YamlLoader)

    def _load_model(self):
        # TODO: We hardcode the path to the `model.yml`, fix when there is a
        # command to get the model.
        model_path = f"{self._resume_path}/context/model.yml"
        if not os.path.isfile(model_path):
            return
        with open(model_path) as f:
            self._set_model(Binary.deserialize(f))
