#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from ._capi import _api
from .utils import make_python_string


class Kind:
    def __init__(self, kind):
        self._kind = kind

    @property
    def name(self):
        name = _api.rp_kind_get_name(self._kind)
        return make_python_string(name)
