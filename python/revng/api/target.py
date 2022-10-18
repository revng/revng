#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Generator, List, Optional

from ._capi import _api, ffi
from .container import Container
from .kind import Kind
from .utils import make_c_string, make_generator, make_python_string


class Target:
    def __init__(self, target, container: Container):
        self._target = target
        self._container = container

    @staticmethod
    def create(kind: Kind, container: Container, path_components: List[str]) -> Optional["Target"]:
        _path_components = [make_c_string(c) for c in path_components]
        _target = _api.rp_target_create(
            kind._kind,
            len(_path_components),
            _path_components,
        )

        return Target(_target, container) if _target != ffi.NULL else None

    @property
    def kind(self) -> Optional[Kind]:
        _kind = _api.rp_target_get_kind(self._target)
        return Kind(_kind) if _kind != ffi.NULL else None

    @property
    def is_ready(self) -> bool:
        return bool(_api.rp_target_is_ready(self._target, self._container._container))

    @property
    def path_components_count(self) -> int:
        return _api.rp_target_path_components_count(self._target)

    def _get_path_component(self, idx: int) -> str:
        path_component = _api.rp_target_get_path_component(self._target, idx)
        return make_python_string(path_component)

    def path_components(self) -> Generator[str, None, None]:
        for idx in range(self.path_components_count):
            yield self._get_path_component(idx)

    def joined_path(self):
        return "/".join(self.path_components())

    def serialize(self) -> str:
        _serialized = _api.rp_target_create_serialized_string(self._target)
        return make_python_string(_serialized)

    def extract(self) -> str:
        _element = _api.rp_container_extract_one(self._container._container, self._target)
        return make_python_string(_element)

    def as_dict(self):
        return {
            "serialized": self.serialize(),
            "path_components": list(self.path_components()),
            "kind": self.kind.name if self.kind is not None else None,
            "ready": self.is_ready,
        }


class TargetsList:
    def __init__(self, targets_list, container: Container):
        self._targets_list = targets_list
        self._container = container

    def targets(self) -> Generator[Target, None, None]:
        return make_generator(len(self), self._get_nth_target)

    def _get_nth_target(self, idx: int) -> Optional[Target]:
        _target = _api.rp_targets_list_get_target(self._targets_list, idx)
        return Target(_target, self._container) if _target != ffi.NULL else None

    def __len__(self):
        return _api.rp_targets_list_targets_count(self._targets_list)
