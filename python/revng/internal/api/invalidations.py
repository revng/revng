#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

from dataclasses import dataclass
from typing import Generic, TypeVar

from ._capi import _api
from .utils import make_python_string


class Invalidations:
    def __init__(self, invalidations=None):
        if invalidations is None:
            self._invalidations = _api.rp_invalidations_create()
        else:
            self._invalidations = invalidations

    def __str__(self):
        res = _api.rp_invalidations_serialize(self._invalidations)
        return make_python_string(res)


T = TypeVar("T")


@dataclass
class ResultWithInvalidations(Generic[T]):
    result: T
    invalidations: Invalidations
