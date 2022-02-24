#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import Union

from cffi import FFI

ffi = FFI()


def make_c_string(s: Union[bytes, str]):
    if isinstance(s, bytes):
        return ffi.new("char[]", s)
    elif isinstance(s, str):
        return ffi.new("char[]", s.encode("utf-8"))
    else:
        raise TypeError(f"Invalid type: {s.__class__}")


def make_python_string(s: ffi.CData):
    bytestring = ffi.string(s)
    return bytestring.decode("utf-8")
