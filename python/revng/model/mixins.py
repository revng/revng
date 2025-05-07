#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Mapping


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
        Fetch the artifacts from the `Binary`.
        """
        result = self._project.get_artifact(artifact_name, [self])  # type: ignore[attr-defined]
        if isinstance(result, Mapping):
            keys = list(result.keys())
            assert len(keys) == 1
            return result[keys[0]]
        else:
            return result


BinaryMixin = _ArtifactMixin
FunctionMixin = _ArtifactMixin
TypeDefinitionMixin = _ArtifactMixin
