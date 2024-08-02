#
# This file is distributed under the MIT License. See LICENSE.md for details.
#


class AllMixin:
    _project = None


class BinaryMixin(AllMixin):
    def get_artifact(self, artifact_name: str) -> bytes | str:
        return self._project._get_artifact(artifact_name, [self])  # type: ignore[attr-defined]


class FunctionMixin(AllMixin):
    def get_artifact(self, artifact_name: str) -> bytes | str:
        return self._project._get_artifact(artifact_name, [self])  # type: ignore[attr-defined]


class TypeDefinitionMixin(AllMixin):
    def get_artifact(self, artifact_name: str) -> bytes | str:
        return self._project._get_artifact(artifact_name, [self])  # type: ignore[attr-defined]
