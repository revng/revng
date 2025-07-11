#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Dict, Iterable, List, Mapping, Optional, TypeAlias, Union

from revng.support.artifacts import Artifact
from revng.tupletree import StructBase

ArtifactResult: TypeAlias = Union[Artifact, Mapping[str, Artifact]]


class AllMixin:
    """
    Base mixin for all the objects in the `Binary` that don't support getting the artifacts.
    """

    _project = None
    _keys = None


class _ArtifactMixin(AllMixin):
    """
    Mixin used to implement the get_artifact function
    """

    def get_artifact(self, artifact_name: str):
        """
        Fetch the artifacts that belong to the current model entity
        """
        result = self._project._get_artifact(artifact_name, [self])  # type: ignore[attr-defined]
        if isinstance(result, Mapping):
            keys = list(result.keys())
            assert len(keys) == 1
            return result[keys[0]]
        else:
            return result


class BinaryMixin(_ArtifactMixin):
    def get_artifact(self, artifact_name: str, targets=None):
        """
        Fetch the artifacts from the `Binary`.
        """
        if targets is None:
            return self._project._get_artifact(artifact_name, [self])  # type: ignore[attr-defined]
        else:
            return self._project._get_artifact(artifact_name, targets)  # type: ignore[attr-defined]

    def commit(self):
        """
        Persist the changes to the backend.
        """
        self._project._commit()  # type: ignore[attr-defined]

    def revert(self):
        """
        Revert changes made since the last call to `commit()`.
        """
        self._project._revert()  # type: ignore[attr-defined]

    def get_artifacts(
        self, params: Dict[str, Optional[Iterable[Union[str, StructBase]]]]
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
        return self._project._get_artifacts(params)  # type: ignore[attr-defined]

    def analyze(
        self, analysis_name: str, targets: Dict[str, List[str]] = {}, options: Dict[str, str] = {}
    ):
        """
        Run a single analysis. In addition to the `analysis_name` you need to
        specify a dict of targets, some analysis require you to also pass an
        `options` dict.
        """
        return self._project._analyze(analysis_name, targets, options)  # type: ignore[attr-defined]

    def analyses_list(self, analysis_name: str):
        """
        Run analysis list, these are predefined list of analysis that run sequentially.
        """
        return self._project._analyses_list(analysis_name)  # type: ignore[attr-defined]


FunctionMixin = _ArtifactMixin
TypeDefinitionMixin = _ArtifactMixin
