#
# This file is distributed under the MIT License.See LICENSE.md for details.
#

from typing import Generator, Optional

from ._capi import _api, ffi
from .utils import make_c_string, make_generator, make_python_string


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
    def doc(self) -> str:
        _doc = _api.rp_rank_get_doc(self._rank)
        return make_python_string(_doc)

    @property
    def depth(self) -> int:
        return _api.rp_rank_get_depth(self._rank)

    # WIP: property
    def get_parent(self) -> Optional["Rank"]:
        _rank = _api.rp_rank_get_parent(self._rank)
        return Rank(_rank) if _rank != ffi.NULL else None

    def as_dict(self):
        ret = {"name": self.name, "doc": self.doc, "depth": self.depth}
        parent = self.get_parent()

        if parent is not None:
            ret["parent"] = parent.name
        return ret

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash((self.name,))
