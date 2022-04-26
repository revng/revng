#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import traceback
from pathlib import Path
from typing import List

from cffi import FFI
from cffi.backend_ctypes import CTypesBackend

# We assume this file is in <root>/lib/python/revng/api
ROOT = (Path(__file__) / "../../../../..").resolve()
HEADERS_DIR = ROOT / "include" / "revng" / "PipelineC"

header_paths = [
    HEADERS_DIR / "ForwardDeclarationsC.h",
    HEADERS_DIR / "Prototypes.h",
]

ffi = FFI()
for header_path in header_paths:
    with open(header_path, encoding="utf-8") as header_file:
        lines = []
        for line in header_file:
            if not line.startswith("#"):
                lines.append(line)
        ffi.cdef("\n".join(lines))


@ffi.callback("void ()")
def pytraceback():
    traceback.print_stack()


LIBRARY_PATH = ROOT / "lib" / "librevngPipelineC.so"
assert LIBRARY_PATH.is_file()

ARGV: List[str] = []

REVNG_ANALYSIS_LIBRARIES = os.getenv("REVNG_ANALYSIS_LIBRARIES", "")
ANALYSIS_LIBRARIES = [Path(f) for f in REVNG_ANALYSIS_LIBRARIES.split(":")]

assert all(lib.is_file() for lib in ANALYSIS_LIBRARIES), "Analysis libraries must all be present"


_ANALYSIS_LIBRARIES = [
    ffi.new("char[]", str(s.resolve()).encode("utf-8")) for s in ANALYSIS_LIBRARIES
]

ctypes_backend = CTypesBackend()

_api = ffi.dlopen(str(LIBRARY_PATH), flags=ctypes_backend.RTLD_NOW | ctypes_backend.RTLD_GLOBAL)
_api.rp_set_custom_abort_hook(pytraceback)


def initialize():
    """Initialize library, must be called exactly once"""
    success = _api.rp_initialize(len(ARGV), ARGV, len(_ANALYSIS_LIBRARIES), _ANALYSIS_LIBRARIES)
    assert success, "Failed revng C API initialization"
