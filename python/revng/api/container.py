#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Dict

from ._capi import _api
from .utils import make_c_string, make_python_string


class ContainerIdentifier:
    def __init__(self, container_identifier):
        self._container_identifier = container_identifier

    @property
    def name(self) -> str:
        name = _api.rp_container_identifier_get_name(self._container_identifier)
        return make_python_string(name)


class Container:
    def __init__(self, container, step_name):
        self._container = container
        self._step_name = step_name

    @property
    def name(self) -> str:
        name = _api.rp_container_get_name(self._container)
        return make_python_string(name)

    @property
    def mime(self) -> str:
        mime = _api.rp_container_get_mime(self._container)
        return make_python_string(mime)

    def store(self, path: str) -> bool:
        _path = make_c_string(path)
        return _api.rp_container_store(self._container, _path)

    def load(self, path: str) -> bool:
        _path = make_c_string(path)
        return _api.rp_container_load(self._container, _path)

    def as_dict(self) -> Dict[str, str]:
        return {"name": self.name, "mime": self.mime, "_step": self._step_name}

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._container == self._container

    def __hash__(self):
        return hash(self._container)
