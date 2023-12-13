#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Mapping

from ._capi import _api
from .utils import make_c_string


class StringMap:
    def __init__(self, content: Mapping[str, str] | None = None):
        self._string_map = _api.rp_string_map_create()

        if content is not None:
            for key, value in content.items():
                self.add(key, value)

    def add(self, key: str, value: str):
        _api.rp_string_map_insert(self._string_map, make_c_string(key), make_c_string(value))
