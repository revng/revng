#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Union, TypeVar

import revng.support.artifacts as artifacts
from revng.project.common import AllObjects, ALL_OBJECTS
from revng.support import IgnoreDeepCopy
from revng.support.artifacts import Artifact
from revng.tupletree import StructBase

StructBaseT = TypeVar("StructBaseT", bound=StructBase)

def _only(dict_: Mapping[str, Artifact]) -> Artifact:
    assert isinstance(dict_, dict)
    assert len(dict_) == 1
    return dict_[list(dict_)[0]]


class AllMixin:
    """
    Base mixin for all the objects in the `Binary` that don't support getting the artifacts.
    """

    _project = None
    _location = None


class BinaryMixin(AllMixin):
    @property
    def _location(self):
        return "/binary"

    def get_artifact(
        self,
        artifact_name: str,
        objects: Union[Sequence[Union[str, StructBaseT]], AllObjects] = ALL_OBJECTS,
    ) -> Mapping[str, Artifact]:
        """
        Fetch the artifacts from the `Binary`.
        """
        return self._project.get()._get_artifact(artifact_name, objects)  # type: ignore[attr-defined]

    def commit(self):
        """
        Persist the changes to the backend.
        """
        self._project.get()._commit()  # type: ignore[attr-defined]

    def revert(self):
        """
        Revert changes made since the last call to `commit()`.
        """
        self._project.get()._revert()  # type: ignore[attr-defined]

    def analyze(
        self,
        analysis_name: str,
        configuration: Dict[str, str] = {},
        containers: Dict[str, List[str]] = {},
    ):
        """
        Run a single analysis or an analysis list. In addition to the
        `analysis_name` you can optionally specify a dict of objects in
        `containers`. Some analyses require a configuration, which can be
        passed in the `configuration` dict.
        """
        return self._project.get()._analyze(analysis_name, configuration, containers)

{% for artifact in binary_artifacts %}
    @property
    def {{ artifact.name | normalize }}(self) -> artifacts.{{ artifact.type_ }}:
        return _only(self.get_artifact("{{ artifact.name }}"))
{% endfor %}


class _ArtifactMixin(AllMixin):
    """
    Mixin used to implement the get_artifact function
    """

    def get_artifact(self, name: str) -> Artifact:
        """
        Fetch the artifacts that belong to the current model entity
        """
        return _only(self._project.get()._get_artifact(name, [self]))  # type: ignore[attr-defined]


class FunctionMixin(_ArtifactMixin):
    @property
    def _location(self):
        return f"/function/{self.key()}"

{% for artifact in function_artifacts %}
    @property
    def {{ artifact.name | normalize }}(self) -> artifacts.{{ artifact.type_ }}:
        return self.get_artifact("{{ artifact.name }}")
{% endfor %}


class TypeDefinitionMixin(_ArtifactMixin):
    @property
    def _location(self):
        return f"/type-definition/{self.key()}"

{% for artifact in typedefinition_artifacts %}
    @property
    def {{ artifact.name | normalize }}(self) -> artifacts.{{ artifact.type_ }}:
        return self.get_artifact("{{ artifact.name }}")
{% endfor %}
