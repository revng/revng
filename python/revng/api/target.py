#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import List, Optional, Generator

from ._capi import _api, ffi
from .kind import Kind
from .utils import make_c_string, make_python_string, make_generator


class Target:
    def __init__(self, target):
        self._target = target

    @staticmethod
    def create(kind: Kind, exact: bool, path_components: List[str]) -> Optional["Target"]:
        _path_components = [make_c_string(c) for c in path_components]
        _target = ffi.gc(
            _api.rp_target_create(
                kind._kind,
                int(exact),
                len(_path_components),
                _path_components,
            ),
            _api.rp_target_destroy,
        )
        return Target(_target) if _target != ffi.NULL else None

    @property
    def kind(self) -> Optional[Kind]:
        _kind = _api.rp_target_get_kind(self._target)
        return Kind(_kind) if _kind != ffi.NULL else None

    @property
    def is_exact(self) -> bool:
        return bool(_api.rp_target_is_exact(self._target))

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
        serialized = make_python_string(_serialized, True)
        return serialized

    def as_dict(self):
        return {
            "serialized": self.serialize(),
            "exact": self.is_exact,
            "path_components": list(self.path_components()),
            "kind": self.kind.name,
            "ready": True,
        }


class TargetsList:
    # TODO: maybe we want to have __iter__ and __next__

    def __init__(self, targets_list):
        self._targets_list = targets_list

    def targets(self) -> Generator[Target, None, None]:
        return make_generator(len(self), self._get_nth_target)

    def _get_nth_target(self, idx: int) -> Optional[Target]:
        _target = _api.rp_targets_list_get_target(self._targets_list, idx)
        return Target(_target) if _target != ffi.NULL else None

    def __len__(self):
        return _api.rp_targets_list_targets_count(self._targets_list)
