#
# This file is distributed under the MIT License.See LICENSE.md for details.
#

from typing import Optional, Generator
from ._capi import _api, ffi
from .utils import make_python_string, make_c_string, make_generator


class Rank:
    @staticmethod
    def ranks() -> Generator["Rank", None, None]:
        return make_generator(Rank._get_count(), Rank._get_idx)

    @staticmethod
    def from_name(name: str) -> Optional["Rank"]:
        c_name = make_c_string(name)
        _rank = _api.rp_rank_get_from_name(c_name)
        return Rank(_rank) if _rank != ffi.NULL else None

    @staticmethod
    def _get_count() -> int:
        return _api.rp_ranks_count()

    @staticmethod
    def _get_idx(idx: int) -> Optional["Rank"]:
        _rank = _api.rp_rank_get(idx)
        return Rank(_rank) if _rank != ffi.NULL else None

    def __init__(self, rank):
        self._rank = rank

    @property
    def name(self) -> str:
        _name = _api.rp_rank_get_name(self._rank)
        return make_python_string(_name)

    @property
    def depth(self) -> int:
        return _api.rp_rank_get_depth(self._rank)

    def get_parent(self) -> Optional["Rank"]:
        _rank = _api.rp_rank_get_parent(self._rank)
        return Rank(_rank) if _rank != ffi.NULL else None

    def as_dict(self):
        ret = {"name": self.name, "depth": self.depth}
        parent = self.get_parent()

        if parent is not None:
            ret["parent"] = parent.name
        return ret
