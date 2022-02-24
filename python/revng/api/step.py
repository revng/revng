#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from ._capi import _api
from .container import ContainerIdentifier, Container
from .utils import make_python_string


class Step:
    def __init__(self, step):
        self._step = step

    @property
    def name(self):
        name = _api.rp_step_get_name(self._step)
        return make_python_string(name)

    def get_container(self, container_identifier: ContainerIdentifier):
        container = _api.rp_step_get_container(self._step, container_identifier._container_identifier)
        return Container(container)

    def _get_container(self, container_identifier: ContainerIdentifier):
        container = _api.rp_step_get_container(self._step, container_identifier._container_identifier)
        return Container(container)
