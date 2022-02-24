#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import List, Optional

from ._capi import _api
from .kind import Kind
from .utils import make_c_string, make_python_string


class Target:
    def __init__(self, target, _free_on_delete=False):
        self._target = target
        self._free_on_delete = _free_on_delete

    @staticmethod
    def create(kind: Kind, exact: bool, path_components: List[str]) -> Optional["Target"]:
        _path_components = [make_c_string(c) for c in path_components]
        target = _api.rp_target_create(
            kind._kind,
            int(exact),
            len(_path_components),
            _path_components,
        )
        if not target:
            return None

        return Target(target, _free_on_delete=True)

    @property
    def kind(self):
        kind = _api.rp_target_get_kind(self._target)
        return Kind(kind)

    @property
    def is_exact(self):
        return bool(_api.rp_target_is_exact(self._target))

    @property
    def path_components_count(self):
        return _api.rp_target_path_components_count(self._target)

    def _get_path_component(self, idx: int):
        path_component = _api.rp_target_get_path_component(self._target, idx)
        return make_python_string(path_component)

    def path_components(self):
        for idx in range(self.path_components_count):
            yield self._get_path_component(idx)

    def serialize(self) -> str:
        _serialized = _api.rp_target_create_serialized_string(self._target)
        serialized = make_python_string(_serialized)
        _api.rp_string_destroy(_serialized)
        return serialized

    def __del__(self):
        if self._free_on_delete:
            _api.rp_target_destroy(self._target)
