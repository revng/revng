#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from pathlib import Path
from typing import List

from cffi import FFI
from cffi.backend_ctypes import CTypesBackend


ORCHESTRA_ROOT = Path(__file__).parent.parent.parent.parent.parent
HEADERS_DIR = ORCHESTRA_ROOT / "include" / "revng" / "PipelineC"

header_paths = [
    HEADERS_DIR / "c_typedefs.h",
    HEADERS_DIR / "prototypes.h",
]

ffi = FFI()
for header_path in header_paths:
    with open(header_path) as header_file:
        lines = []
        for line in header_file:
            if not line.startswith("#"):
                lines.append(line)
        ffi.cdef("\n".join(lines))

LIB_DIR = ORCHESTRA_ROOT / "lib"
LIBRARY_PATH = str(LIB_DIR / "librevngPipelineC.so")

ARGV: List[str] = []
ANALYSIS_LIBRARIES = [
    str(LIB_DIR / "revng/analyses/librevngFunctionIsolation.so"),
]

_ANALYSIS_LIBRARIES = [ffi.new("char[]", s.encode("utf-8")) for s in ANALYSIS_LIBRARIES]


ctypes_backend = CTypesBackend()

_api = ffi.dlopen(LIBRARY_PATH, flags=ctypes_backend.RTLD_NOW | ctypes_backend.RTLD_GLOBAL)
if not _api.rp_is_initialized():
    success = _api.rp_initialize(len(ARGV), ARGV, len(_ANALYSIS_LIBRARIES), _ANALYSIS_LIBRARIES)
    assert success, "Failed revng C API initialization"