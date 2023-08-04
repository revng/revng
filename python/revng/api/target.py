#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from collections.abc import Sequence
from itertools import islice
from typing import Any, Generator, List, Optional

from ._capi import _api, ffi
from .utils import convert_buffer, make_c_string, make_python_string


class Target:
    def __init__(self, target, container):
        self._target = target
        self._container = container

    @staticmethod
    def create(_kind, container: str, path_components: List[str]) -> Optional["Target"]:
        _path_components = [make_c_string(c) for c in path_components]
        _target = _api.rp_target_create(
            _kind,
            len(_path_components),
            _path_components,
        )

        return Target(_target, container) if _target != ffi.NULL else None

    @property
    def kind(self) -> str | None:
        _kind_name = _api.rp_target_get_kind(self._target)
        return make_python_string(_kind_name) if _kind_name != ffi.NULL else None

    @property
    def is_ready(self) -> bool:
        return bool(_api.rp_target_is_ready(self._target, self._container))

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

    def extract(self) -> str | bytes | None:
        _buffer = _api.rp_container_extract_one(self._container, self._target)
        if _buffer == ffi.NULL:
            return None
        size = _api.rp_buffer_size(_buffer)
        data = _api.rp_buffer_data(_buffer)
        _mime = _api.rp_container_get_mime(self._container)
        return convert_buffer(data, size, make_python_string(_mime))

    def as_dict(self):
        return {
            "serialized": self.serialize(),
            "pathComponents": list(self.path_components()),
            "kind": self.kind,
            "ready": self.is_ready,
        }


class TargetsList(Sequence):
    def __init__(self, targets_list, container):
        self._targets_list = targets_list
        self._container = container

    def _get_target(self, idx: int) -> Optional[Target]:
        _target = _api.rp_targets_list_get_target(self._targets_list, idx)
        return Target(_target, self._container) if _target != ffi.NULL else None

    def __len__(self) -> int:
        return _api.rp_targets_list_targets_count(self._targets_list)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            if idx < len(self):
                return self._get_target(idx)
            else:
                raise IndexError("list index out of range")
        else:
            return list(islice(self, idx.start, idx.stop, idx.step))


class ContainerToTargetsMap:
    def __init__(self, _map: Any | None = None):
        if _map is None:
            self._map = _api.rp_container_targets_map_create()
        else:
            self._map = _map

    def add(self, _container, target: Target):
        _api.rp_container_targets_map_add(self._map, _container, target._target)
