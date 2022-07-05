#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from collections.abc import Sequence
from typing import Any, Callable, Generic, Optional, TypeVar

from ._capi import _api, ffi
from .exceptions import RevngPipelineCException
from .utils import make_python_string


class ErrorList(Sequence):
    def __init__(self, _error_list: Optional[Any] = None):
        if _error_list is not None:
            self._error_list = _error_list
        else:
            self._error_list = _api.rp_make_error_list()

    def empty(self) -> bool:
        return _api.rp_error_list_is_empty(self._error_list)

    def _get_message(self, index: int) -> Optional[str]:
        result = _api.rp_error_list_get_error_message(self._error_list, index)
        return make_python_string(result) if result != ffi.NULL else None

    def get_last_message(self) -> Optional[str]:
        if not self.empty():
            return self[-1]
        return None

    def check(self):
        if not self.empty():
            raise RevngPipelineCException(self.get_last_message())

    def __len__(self) -> int:
        return _api.rp_error_list_size(self._error_list)

    def __getitem__(self, index) -> str:
        if (result := self._get_message(index)) is None:
            raise IndexError("Invalid index")
        return result


T = TypeVar("T")


class Expected(Generic[T]):
    def __init__(self, result: Any, error_list: ErrorList):
        self.result: Optional[T] = result if result != ffi.NULL else None
        self.error_list = error_list

    def __bool__(self):
        return self.error_list.empty()

    def check(self):
        return self.error_list.empty() and bool(self.result)


def run_with_el(func: Callable[..., T], *args) -> Expected[T]:
    el = ErrorList()
    result = func(*args, el._error_list)
    return Expected(result, el)
