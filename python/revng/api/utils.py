#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import TypeVar, Callable, Generator, Optional

from ._capi import _api, ffi, make_c_string


make_c_string = make_c_string


def make_python_string(s: ffi.CData, free: bool = False):  # type: ignore
    bytestring = ffi.string(s)
    ret = bytestring.decode("utf-8")
    if free:
        _api.rp_string_destroy(s)
    return ret


T = TypeVar("T")


def make_generator(count: int, get_idx: Callable[[int], Optional[T]]) -> Generator[T, None, None]:
    for i in range(count):
        ret = get_idx(i)
        if ret is not None:
            yield ret
