#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import shutil
from abc import ABC, abstractmethod
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import List

from revng.model import Binary, Function, StructBase, get_element_by_path, make_diff  # type: ignore[attr-defined] # noqa: E501
from revng.tupletree import TypedList


class Project(ABC):
    def __init__(self) -> None:
        self.model: Binary = Binary()
        self.last_saved_model: Binary = deepcopy(Binary)
        self.pipeline_description: dict = {}
        self.input_binary: str
        self.revng_executable: str
        self.resume_dir: str

    @abstractmethod
    def import_binary(self, input_binary: str):
        """
        Use this method to import the binary into the project. This method should
        implement how the binary is imported and set the variable `self.input_binary`.
        """
        pass

    def get_artifact(self, artifact_name: str) -> bytes | str:
        """
        High level method that the user can call to fetch an artifact without specifying targets.
        """
        return self._get_artifact(artifact_name)

    @abstractmethod
    def _get_artifact(
        self, artifact_name: str, targets_class: List[StructBase] = []
    ) -> bytes | str:
        """
        Resolves the target logic, performs MIME conversion and dispatches
        the work to the `_get_artifact_internal` to actually get the artifact.
        This method is a more advanced version of `get_artifact` for when you
        want more control of which artifact are processed.
        """
        pass

    @abstractmethod
    def _get_artifact_internal(
        self, artifact_name: str, targets_class: List[str] = []
    ) -> bytes | str:
        """
        The actual implementation of how to fetch the artifacts, eg: call subprocess or gql.
        This is the low-level method that actually calls `revng` and gets the artifact.
        """
        pass

    def get_artifacts(self, params: dict[str, List[StructBase]]) -> dict[str, str | bytes]:
        """
        Allows fetching multiple artifacts at once. The `params` is a dict containing
        the name of the artifact and a list of targets (it can be empty to fetch all the
        targets).
        Example `params`:
        params = {
            "disassemble": [Function_1, Function_2],
            "decompile": []
        }
        """
        result: dict[str, str | bytes] = {}
        for artifact_name, targets in params.items():
            result[artifact_name] = self._get_artifact(artifact_name, targets)

        return result

    @abstractmethod
    def analyze(
        self, analysis_name: str, targets: dict[str, List[str]] = {}, options: dict[str, str] = {}
    ) -> None:
        """
        Run analysis, to get a list of available analysis run (in orchestra)
        `orc shell revng analyze`.
        """
        pass

    @abstractmethod
    def analyses_list(self, analysis_name: str) -> None:
        """
        Run analysis list, these are predefined list of analysis that run sequentially,
        to get a list of available analysis run (in orchestra) `orc shell revng analyze`.
        """
        pass

    @abstractmethod
    def commit(self) -> None:
        """
        Persist the changes from `self.model` to the backend. You need to call
        this method on every change you make to the model.
        """
        pass

    def revert(self) -> None:
        """
        Revert changes made to the `self.model`. Note that this doesn't revert
        changes that you committed with `self.commit()`.
        """
        diff = make_diff(self.model, self.last_saved_model)
        if len(diff.changes) == 0:
            return
        self._set_model(deepcopy(self.last_saved_model))

    @abstractmethod
    def _get_pipeline_description(self) -> None:
        """
        Define how to implement getting the pipeline description data, The method should parse the
        data as `yaml` and store it in the variable `self.pipeline_description`
        """
        pass

    def import_and_analyze(self, input_binary_path: str):
        """
        A helper method that does everything you usually want to do
        when you start working on a project.
        """
        self.import_binary(input_binary_path)
        self.analyses_list("revng-initial-auto-analysis")
        self.analyses_list("revng-c-initial-auto-analysis")
        self._get_pipeline_description()
        self._set_model_mixins()

    def get_targets(self, analysis_name: str, targets: List[Function]) -> dict[str, List[str]]:
        """
        Create a dict of targets from the container input and the list `Function` objects
        """
        container_input = self._get_analysis_inputs(analysis_name)

        _targets: dict[str, List[str]] = {}
        for input, kinds in container_input.items():  # noqa: A001
            _targets[input] = []
            for kind in kinds:
                if kind == "root":
                    _targets[input].append(f":{kind}")
                else:
                    _targets[input].extend(f"{target.key()}:{kind}" for target in targets)
        return _targets

    def _set_revng_executable(self, revng_executable_path: str | None):
        """
        Check if the path of the user supplied or default revng executable
        is present on the system
        """
        if not revng_executable_path:
            self.revng_executable = "revng"
        else:
            self.revng_executable = revng_executable_path
        assert shutil.which(self.revng_executable) is not None

    def _set_resume_dir(self, resume_dir: str | None):
        """
        Use the user supplied resume dir or create a temp dir
        """
        if not resume_dir:
            self._tmp_dir = TemporaryDirectory()
            self.resume_dir = self._tmp_dir.name
        else:
            self.resume_dir = resume_dir

    def _set_model(self, model: Binary) -> None:
        self.model = model
        self.last_saved_model = deepcopy(self.model)
        # We call this method in multiple places, during the initial auto
        # analysis for example we don't have the pipeline description yet
        # so we need to take that into account.
        if self.pipeline_description:
            self._set_model_mixins()

    def _set_model_mixins(self) -> None:
        for ct in self.pipeline_description["Ranks"]:
            model_path = ct.get("ModelPath")
            if not model_path:
                continue

            obj = get_element_by_path(model_path, self.model)
            if isinstance(obj, TypedList):
                for elem in obj:
                    elem._project = self
            else:
                obj._project = self

    def _get_analysis_inputs(self, analysis_name: str) -> dict[str, List[str]]:
        inputs: dict[str, List[str]] = {}
        for step in self.pipeline_description["Steps"]:
            for analysis in step["Analyses"]:
                if analysis["Name"] == analysis_name:
                    for i in analysis["ContainerInputs"]:
                        inputs[i["Name"]] = i["AcceptableKinds"]
        return inputs

    def _get_artifact_kind(self, step_name: str) -> str:
        for step in self.pipeline_description["Steps"]:
            if step["Name"] == step_name:
                return step["Artifacts"]["Kind"]
        raise RuntimeError(f"Couldn't find step: {step_name}")

    def _get_artifact_container(self, step_name) -> str:
        for step in self.pipeline_description["Steps"]:
            if step["Name"] == step_name:
                return step["Artifacts"]["Container"]
        raise RuntimeError(f"Couldn't find step: {step_name}")

    def _get_step_name(self, analysis_name: str) -> str:
        for step in self.pipeline_description["Steps"]:
            for analysis in step["Analyses"]:
                if analysis["Name"] == analysis_name:
                    return step["Name"]
        raise RuntimeError(f"Couldn't find step for analysis: {analysis_name}")

    def _get_analyses_list_names(self) -> List[str]:
        analyses_names = []
        for analyses in self.pipeline_description["AnalysesLists"]:
            analyses_names.append(analyses["Name"])
        return analyses_names

    def _get_result_mime(self, artifact_name: str, result: str | bytes) -> bytes | str:
        container_mime = ""
        artifact_container = self._get_artifact_container(artifact_name)
        for container in self.pipeline_description["Containers"]:
            if container["Name"] == artifact_container:
                container_mime = container["MIMEType"]

        if not container_mime:
            raise RuntimeError(f"Couldn't find step name {artifact_name}")

        if not container_mime.endswith("tar+gz") and (
            container_mime.startswith("text") or container_mime == "image/svg"
        ):
            assert isinstance(result, bytes)
            return result.decode("utf-8")
        else:
            return result
