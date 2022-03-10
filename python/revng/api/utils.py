#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import TypeVar, Callable, Generator, Optional, Union

from ._capi import _api, ffi


def make_python_string(s: ffi.CData, free: bool = False) -> str:  # type: ignore
    if s == ffi.NULL:
        return ""
    bytestring = ffi.string(s)
    ret = bytestring.decode("utf-8")
    if free:
        _api.rp_string_destroy(s)
    return ret


def make_c_string(s: Union[bytes, str]):
    if isinstance(s, bytes):
        return ffi.new("char[]", s)
    elif isinstance(s, str):
        return ffi.new("char[]", s.encode("utf-8"))
    else:
        raise TypeError(f"Invalid type: {s.__class__}")


T = TypeVar("T")


def make_generator(count: int, get_idx: Callable[[int], Optional[T]]) -> Generator[T, None, None]:
    for i in range(count):
        ret = get_idx(i)
        if ret is not None:
            yield ret
