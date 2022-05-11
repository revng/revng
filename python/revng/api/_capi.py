#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import re
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List

from cffi import FFI
from cffi.backend_ctypes import CTypesBackend


class ApiWrapper:
    function_matcher = re.compile(
        r"(?P<return_type>[\w_]+)\s*\*\s*\/\*\s*owning\s*\*\/\s*(?P<function_name>[\w_]+)",
        re.M | re.S,
    )

    def __init__(self, api, ffi: FFI):
        self.__api = api
        self.__ffi = ffi
        self.__proxy: Dict[str, Callable[..., Any]] = {}

        for match in self.function_matcher.finditer("\n".join(ffi._cdefsources)):
            function_name = match.groupdict()["function_name"]
            return_type = match.groupdict()["return_type"]
            function = getattr(self.__api, function_name)
            if return_type == "char":
                destructor = self.__api.rp_string_destroy
            elif hasattr(self.__api, f"{return_type}_destroy"):
                destructor = getattr(self.__api, f"{return_type}_destroy")
            else:
                raise ValueError(f"Could not find suitable destructor for {return_type}")

            self.__proxy[function_name] = self.__wrap_gc(function, destructor)

    @staticmethod
    def __wrap_gc(function, destructor):
        @wraps(function)
        def new_function(*args):
            ret = function(*args)
            return ffi.gc(ret, destructor)

        return new_function

    def __getattr__(self, name: str):
        if name in self.__proxy:
            return self.__proxy[name]
        else:
            return getattr(self.__api, name)

    def __setattr__(self, name, value):
        if re.search(r"__\w*$", name):
            self.__dict__[name] = value
        else:
            raise NotImplementedError()


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

_raw_api = ffi.dlopen(str(LIBRARY_PATH), flags=ctypes_backend.RTLD_NOW | ctypes_backend.RTLD_GLOBAL)
_api = ApiWrapper(_raw_api, ffi)
_api.rp_set_custom_abort_hook(pytraceback)


def initialize():
    """Initialize library, must be called exactly once"""
    success = _api.rp_initialize(len(ARGV), ARGV, len(_ANALYSIS_LIBRARIES), _ANALYSIS_LIBRARIES)
    assert success, "Failed revng C API initialization"
