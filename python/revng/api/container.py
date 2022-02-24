#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from ._capi import _api
from .target import Target
from .utils import make_c_string, make_python_string


class ContainerIdentifier:
    def __init__(self, container_identifier):
        self._container_identifier = container_identifier

    @property
    def name(self):
        name = _api.rp_container_identifier_get_name(self._container_identifier)
        return make_python_string(name)


class TargetsList:
    # TODO: maybe we want to have __iter__ and __next__

    def __init__(self, targets_list):
        self._targets_list = targets_list

    def targets(self):
        for idx in range(len(self)):
            yield self._get_nth_target(idx)

    def _get_nth_target(self, idx: int):
        target = _api.rp_targets_list_get_target(self._targets_list, idx)

        if not target:
            return None

        return Target(target)

    def __len__(self):
        return _api.rp_targets_list_targets_count(self._targets_list)


class Container:
    def __init__(self, container):
        self._container = container

    @property
    def name(self) -> str:
        name = _api.rp_container_get_name(self._container)
        return make_python_string(name)

    def store(self, path: str) -> bool:
        _path = make_c_string(path)
        return _api.rp_container_store(self._container, _path)

    def load(self, path: str) -> bool:
        _path = make_c_string(path)
        return _api.rp_container_load(self._container, _path)
