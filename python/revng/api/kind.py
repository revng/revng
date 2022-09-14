#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Any, Dict, Generator, Optional

from ._capi import _api, ffi
from .rank import Rank
from .utils import make_generator, make_python_string


class Kind:
    def __init__(self, kind):
        self._kind = kind

    @property
    def name(self) -> str:
        name = _api.rp_kind_get_name(self._kind)
        return make_python_string(name)

    @property
    def doc(self) -> str:
        doc = _api.rp_kind_get_doc(self._kind)
        return make_python_string(doc)

    @property
    def rank(self) -> Optional[Rank]:
        _rank = _api.rp_kind_get_rank(self._kind)
        return Rank(_rank) if _rank != ffi.NULL else None

    # WIP: not optional
    def get_parent(self) -> Optional["Kind"]:
        _kind = _api.rp_kind_get_parent(self._kind)
        return Kind(_kind) if _kind != ffi.NULL else None

    @property
    def _defined_location_count(self) -> int:
        return _api.rp_kind_get_defined_location_count(self._kind)

    def _get_defined_location_from_index(self, idx: int) -> Optional[Rank]:
        _rank = _api.rp_kind_get_defined_location(self._kind, idx)
        return Rank(_rank) if _rank != ffi.NULL else None

    def defined_locations(self) -> Generator[Rank, None, None]:
        return make_generator(self._defined_location_count, self._get_defined_location_from_index)

    @property
    def _preferred_kind_count(self) -> int:
        return _api.rp_kind_get_preferred_kind_count(self._kind)

    def _get_preferred_kind_from_index(self, idx: int) -> Optional["Kind"]:
        _kind = _api.rp_kind_get_preferred_kind(self._kind, idx)
        return Kind(_kind) if _kind != ffi.NULL else None

    def preferred_kinds(self) -> Generator["Kind", None, None]:
        return make_generator(self._preferred_kind_count, self._get_preferred_kind_from_index)

    def as_dict(self) -> Dict[str, Any]:
        ret = {
            "name": self.name,
            "definedLocations": list(self.defined_locations()),
            "preferredKinds": [k.name for k in self.preferred_kinds()],
        }

        rank = self.rank
        if rank is not None:
            ret["rank"] = rank.name

        parent = self.get_parent()
        if parent is not None:
            ret["parent"] = parent.name

        return ret
