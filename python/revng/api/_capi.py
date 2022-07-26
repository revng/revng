#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
import traceback
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Optional

from cffi import FFI
from cffi.backend_ctypes import CTypesBackend

from revng.support import AnyPaths, to_iterable
from revng.support.collect import collect_libraries, collect_one


class ApiWrapper:
    function_matcher = re.compile(
        r"(?P<return_type>[\w_]+)\s*\*\s*\/\*\s*owning\s*\*\/\s*(?P<function_name>[\w_]+)",
        re.M | re.S,
    )

    def __init__(self, api, ffi: FFI):
        self.__api = api
        self.__ffi = ffi
        self.__lock = Lock()
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

            self.__proxy[function_name] = self.__wrap_lock(self.__wrap_gc(function, destructor))

        for attribute_name in dir(self.__api):
            if attribute_name.startswith("RP_") or attribute_name in self.__proxy:
                continue
            function = getattr(self.__api, attribute_name)
            self.__proxy[attribute_name] = self.__wrap_lock(function)

    def __wrap_gc(self, function, destructor):
        @wraps(function)
        def new_function(*args):
            ret = function(*args)
            if ret != ffi.NULL:
                return ffi.gc(ret, self.__wrap_lock(destructor))
            else:
                return ret

        return new_function

    def __wrap_lock(self, function):
        @wraps(function)
        def new_function(*args):
            with self.__lock:
                return function(*args)

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
    collect_one(ROOT, ["include", "revng", "PipelineC"], "ForwardDeclarationsC.h"),
    collect_one(ROOT, ["include", "revng", "PipelineC"], "Prototypes.h"),
]

ffi = FFI()
for header_path in header_paths:
    assert header_path is not None, "Missing header file"
    with open(header_path, encoding="utf-8") as header_file:
        lines = []
        for line in header_file:
            if not line.startswith("#"):
                lines.append(line)
        ffi.cdef("\n".join(lines))


@ffi.callback("void ()")
def pytraceback():
    traceback.print_stack()


LIBRARY_PATH = collect_one(ROOT, ["lib"], "librevngPipelineC.so")
assert LIBRARY_PATH is not None, "librevngPipelineC.so not found"

ctypes_backend = CTypesBackend()

_raw_api = ffi.dlopen(str(LIBRARY_PATH), flags=ctypes_backend.RTLD_NOW | ctypes_backend.RTLD_GLOBAL)
_api = ApiWrapper(_raw_api, ffi)
_api.rp_set_custom_abort_hook(pytraceback)


def initialize(args: Iterable[str] = (), libraries: Optional[AnyPaths] = None):
    """Initialize library, must be called exactly once"""

    if libraries is None:
        libraries, _ = collect_libraries(ROOT)
    else:
        libraries = to_iterable(libraries)

    path_libraries = [Path(f) for f in libraries]
    assert all(lib.is_file() for lib in path_libraries), "Analysis libraries must all be present"

    _libraries = [ffi.new("char[]", str(s.resolve()).encode("utf-8")) for s in path_libraries]
    _args = [ffi.new("char[]", arg.encode("utf-8")) for arg in args]

    success = _api.rp_initialize(len(_args), _args, len(_libraries), _libraries)
    assert success, "Failed revng C API initialization"


def shutdown():
    _api.rp_shutdown()
