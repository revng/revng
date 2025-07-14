#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import re
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from subprocess import CalledProcessError, run
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, TypeAlias, TypeVar
from typing import Union

from revng.model import Binary, iterate_fields  # type: ignore[attr-defined]
from revng.model.mixins import AllMixin
from revng.pipeline_description import Container, PipelineDescription  # type: ignore[attr-defined]
from revng.support.artifacts import Artifact
from revng.tupletree import StructBase

StructBaseT = TypeVar("StructBaseT", bound=StructBase)
ArtifactResult: TypeAlias = Union[Artifact, Mapping[str, Artifact]]


class Project(ABC):
    """
    This class is the basis for building a project. It defines
    abstract methods that the inherited classes have to implement
    and it also implements some common methods.
    """

    def __init__(self):
        self.model: Binary = Binary()
        self._last_saved_model: Binary = Binary()

        self._pipeline_description: PipelineDescription = self._get_pipeline_description()
        self._rank_matchers: Dict[str, re.Pattern] = self._compute_rank_matchers()
        self._rank_targets: Dict[str, List[str]] = defaultdict(list)
        self._analysis_names = list(
            chain.from_iterable(
                [an.Name for an in st.Analyses] for st in self._pipeline_description.Steps
            )
        )
        self._analyses_list_names = [x.Name for x in self._pipeline_description.AnalysesLists]

    @abstractmethod
    def set_binary_path(self, binary_path: str):
        """
        Use this method to import the binary into the project. This method is a
        prerequisite before running any analyses or getting artifacts.
        """
        pass

    def list_targets(self, artifact_name: str) -> List[str]:
        """Retrieve the list of targets given the artifact"""
        kind_name = self._pipeline_description.Steps[artifact_name].Artifacts.Kind
        rank_name = self._pipeline_description.Kinds[kind_name].Rank
        return self._rank_targets[rank_name]

    def _get_artifact(
        self, artifact_name: str, targets: Optional[Sequence[Union[str, StructBaseT]]] = None
    ) -> ArtifactResult:
        """
        Use the method to fetch one or more targets from an artifact.
        `targets` is a list of targets to produce, each element can be either a
        string which is interpreted as a target string or a model object, which
        will be converted to the corresponding target string.
        If the `targets` parameter is omitted then all targets of the specified
        artifacts are fetched.
        This method automatically performs MIME conversion and dispatches
        the work to the `_get_artifact`.
        The return value can either be a single Artifact object or a dictionary
        of them.
        """
        all_targets = set(self.list_targets(artifact_name))
        if targets is None or (
            len(targets) == 1
            and isinstance(targets[0], StructBase)
            and isinstance(targets[0], AllMixin)
            and targets[0]._keys == []
        ):
            actual_targets = all_targets
        else:
            actual_targets = set()
            for target in targets:
                if isinstance(target, str):
                    actual_targets.add(target)
                elif isinstance(target, StructBase) and isinstance(target, AllMixin):
                    actual_targets.add("/".join(target._keys))
                else:
                    raise ValueError(f"Unknown parameter: {target}")
            assert len(actual_targets) > 0

        assert actual_targets.issubset(all_targets)
        results = self._get_artifact_impl(artifact_name, actual_targets)
        container_mime = self._get_artifact_container(artifact_name).MIMEType

        kind_name = self._pipeline_description.Steps[artifact_name].Artifacts.Kind
        rank_name = self._pipeline_description.Kinds[kind_name].Rank
        rank = self._pipeline_description.Ranks[rank_name]

        if isinstance(results, dict):
            if rank.Depth == 0:
                keys = list(results.keys())
                assert len(keys) == 1
                return Artifact.make(results[keys[0]], container_mime)
            else:
                container_mime = self._mapped_artifact_mime(artifact_name)
                return {k: Artifact.make(v, container_mime) for k, v in results.items()}
        else:
            return Artifact.make(results, container_mime)

    @abstractmethod
    def _get_artifact_impl(
        self, artifact_name: str, targets: Set[str]
    ) -> Union[bytes, Dict[str, bytes]]:
        """
        This method implements the actual fetching of the artifact(s). It is
        implemented by the children of this class.
        """
        pass

    def _mapped_artifact_mime(self, artifact_name: str) -> str:
        """
        Return the artifact's MIME if _get_artifact returns a mapping. By
        default throws an exception. Can be overridden by child classes.
        """
        raise ValueError

    def _get_artifacts(
        self, params: Dict[str, Optional[Sequence[Union[str, StructBase]]]]
    ) -> Dict[str, ArtifactResult]:
        """
        Allows fetching multiple artifacts at once. The `params` is a dict
        containing the name of the artifact and a list of targets (it can be
        empty to fetch all the targets).
        Example `params`:
        params = {
            "disassemble": [Function_1, Function_2],
            "decompile": []
        }
        """
        result: Dict[str, ArtifactResult] = {}
        for artifact_name, targets in params.items():
            result[artifact_name] = self._get_artifact(artifact_name, targets)

        return result

    @abstractmethod
    def _analyze(
        self, analysis_name: str, targets: Dict[str, List[str]] = {}, options: Dict[str, str] = {}
    ):
        """
        Run a single analysis. In addition to the `analysis_name` you need to
        specify a dict of targets, some analysis require you to also pass an
        `options` dict.
        """
        pass

    @abstractmethod
    def _analyses_list(self, analysis_name: str):
        """
        Run analysis list, these are predefined list of analysis that run sequentially.
        """
        pass

    def get_analysis_inputs(self, analysis_name: str) -> Dict[str, List[str]]:
        """
        Get the analysis container inputs and the associate acceptable kinds from
        the `pipeline description`.
        """
        inputs: Dict[str, List[str]] = {}
        for step in self._pipeline_description.Steps:
            if analysis_name in step.Analyses:
                for i in step.Analyses[analysis_name].ContainerInputs:
                    inputs[i.Name] = i.AcceptableKinds
                break

        return inputs

    @abstractmethod
    def _commit(self):
        """
        Persist the changes from `self.model` to the backend.
        """
        pass

    def _revert(self):
        """
        Revert changes made to the `self.model` since the last call to `commit()`.
        """
        self._set_model(deepcopy(self._last_saved_model))

    @abstractmethod
    def _get_pipeline_description(self) -> PipelineDescription:
        """
        Define how to implement getting the pipeline description data. This
        method should parse the data and return a PipelineDescription object.
        """
        pass

    def import_and_analyze(self, binary_path: str):
        """
        A helper method for setting up the project. This method imports
        the binary and runs the `revng` initial auto analysis.
        """
        self.set_binary_path(binary_path)
        self._analyses_list("revng-initial-auto-analysis")

    def _set_model(self, model: Binary):
        self.model = model
        self._last_saved_model = deepcopy(self.model)

        # visit the model, and:
        # * Collect the keys based on _rank_matchers
        # * Fill _project on all classes
        # * Fill _keys where needed and add it to _rank_targets
        new_targets = defaultdict(list)
        for path, obj in iterate_fields(self.model):
            if not isinstance(obj, AllMixin):
                continue

            obj._project = self  # type: ignore
            for rank, pattern in self._rank_matchers.items():
                match = pattern.match(path)
                if match is not None:
                    keys = [k.split("::", 1)[1] if "::" in k else k for k in match.groups()]
                    obj._keys = keys  # type: ignore
                    new_targets[rank].append("/".join(keys))

        self._rank_targets = new_targets

    def _get_step_name(self, analysis_name: str) -> str:
        for step in self._pipeline_description.Steps:
            if analysis_name in step.Analyses:
                return step.Name
        raise RuntimeError(f"Couldn't find step for analysis: {analysis_name}")

    def _get_artifact_container(self, artifact_name: str) -> Container:
        artifact_container = self._pipeline_description.Steps[artifact_name].Artifacts.Container
        container = self._pipeline_description.Containers[artifact_container]
        if container is None:
            raise RuntimeError(f"Couldn't find step name {artifact_name}")
        return container

    def _compute_rank_matchers(self) -> Dict[str, re.Pattern]:
        result = {}
        for rank in self._pipeline_description.Ranks:
            if rank.TupleTreePath == "":
                continue
            path = rank.TupleTreePath
            pattern = "^" + re.sub(r"\\\$\d+", "([^/]*)", re.escape(path)) + "$"
            result[rank.Name] = re.compile(pattern)

        return result


class CLIError(Exception):
    pass


class CLIProjectMixin:
    """Mixin for classes that use the `revng` command-line tool"""

    def __init__(self, revng_executable_path: Optional[str]):
        """
        Check if the path of the user supplied or default revng executable
        is present on the system
        """
        if not revng_executable_path:
            _revng_executable_path = shutil.which("revng")
        else:
            _revng_executable_path = revng_executable_path
        assert _revng_executable_path is not None
        assert os.access(_revng_executable_path, os.F_OK | os.X_OK)
        self._revng_executable_path = _revng_executable_path

    def _run_revng_cli(self, cmd_args: Iterable[str], *args, **kwargs):
        try:
            return run([self._revng_executable_path, *cmd_args], *args, **kwargs, check=True)
        except CalledProcessError as e:
            raise CLIError() from e


class ResumeProjectMixin:
    """Mixin for classes that use a local resume directory"""

    def __init__(self, resume_path: Optional[str]):
        if resume_path is None:
            self._tmp_dir = TemporaryDirectory()
            resume_path = self._tmp_dir.name
        if not os.path.exists(resume_path):
            os.makedirs(resume_path)
        assert os.path.isdir(resume_path)
        self._resume_path = resume_path
