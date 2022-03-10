#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import Optional, Dict

from ._capi import _api, ffi
from .container import ContainerIdentifier, Container
from .utils import make_python_string


class Step:
    def __init__(self, step):
        self._step = step

    @property
    def name(self) -> str:
        name = _api.rp_step_get_name(self._step)
        return make_python_string(name)

    def get_parent(self) -> Optional["Step"]:
        _step = _api.rp_step_get_parent(self._step)
        return Step(_step) if _step != ffi.NULL else None

    def get_container(self, container_identifier: ContainerIdentifier) -> Optional[Container]:
        _container = _api.rp_step_get_container(
            self._step, container_identifier._container_identifier
        )
        return Container(_container) if _container != ffi.NULL else None

    def as_dict(self) -> Dict[str, str]:
        ret = {"name": self.name}

        parent = self.get_parent()
        if parent is not None:
            ret["parent"] = parent.name

        return ret