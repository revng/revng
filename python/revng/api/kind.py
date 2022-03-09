#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import Optional, Dict

from ._capi import _api, ffi
from .rank import Rank
from .utils import make_python_string


class Kind:
    def __init__(self, kind):
        self._kind = kind

    @property
    def name(self) -> str:
        name = _api.rp_kind_get_name(self._kind)
        return make_python_string(name)

    @property
    def rank(self) -> Optional[Rank]:
        _rank = _api.rp_kind_get_rank(self._kind)
        return Rank(_rank) if _rank != ffi.NULL else None

    def get_parent(self) -> Optional["Kind"]:
        _kind = _api.rp_kind_get_parent(self._kind)
        if _kind != ffi.NULL:
            return Kind(_kind)
        else:
            return None

    def as_dict(self) -> Dict[str, str]:
        ret = {"name": self.name}

        rank = self.rank
        if rank is not None:
            ret["rank"] = rank.name

        parent = self.get_parent()
        if parent is not None:
            ret["parent"] = parent.name

        return ret
