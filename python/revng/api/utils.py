#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path

from ._capi import ffi


def make_python_string(s: ffi.CData) -> str:
    if s == ffi.NULL:
        return ""
    bytestring = ffi.string(s)
    return bytestring.decode("utf-8")


def make_c_string(s: bytes | str):
    if isinstance(s, bytes):
        return ffi.new("char[]", s)
    if isinstance(s, str):
        return ffi.new("char[]", s.encode("utf-8"))
    raise TypeError(f"Invalid type: {s.__class__}")


def save_file(path: Path, content: bytes | str):
    if isinstance(content, str):
        with open(path.resolve(), "w", encoding="utf-8") as str_f:
            str_f.write(content)
    else:
        with open(path.resolve(), "wb") as bytes_f:
            bytes_f.write(content)


def convert_buffer(ptr: ffi.CData, size: int, mime: str) -> str | bytes:
    if ptr == ffi.NULL:
        return b""

    data = ffi.unpack(ptr, size)
    if mime.startswith("text/") or mime == "image/svg":
        return data.decode("utf-8")
    else:
        return data
