#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import faulthandler
import re
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Optional

from cffi import FFI
from cffi.backend_ctypes import CTypesBackend

from revng.internal.support import AnyPaths, get_root, to_iterable
from revng.internal.support.collect import collect_libraries, collect_one, collect_pipelines


# This counter is used to count the pointers released by PipelineC, since
# calling rp_shutdown before all pointers are freed leads to a crash
# This will count (atomically) the pointers created by PipelineC and decrements
# once the pointer is freed by `ffi.gc`. Once the API signals that it will no
# longer do anything (by calling `mark_end`) then once the counter reaches 0
# the callback can be invoked which will safely call rp_shutdown.
class AtomicCounterWithCallback:
    """Simple atomic counter, will call callback once mark_end has been called
    and the counter reaches zero"""

    def __init__(self, callback: Callable[[], None]):
        self.lock = Lock()
        self.counter = 0
        self.ending = False
        self.callback = callback

    def increment(self):
        with self.lock:
            self.counter += 1

    def decrement(self):
        with self.lock:
            self.counter -= 1
            if self.counter == 0 and self.ending:
                self.callback()

    def mark_end(self):
        with self.lock:
            self.ending = True
            if self.counter == 0:
                self.callback()


class ApiWrapper:
    function_matcher = re.compile(
        r"(?P<return_type>[\w_]+)\s*\*\s*\/\*\s*owning\s*\*\/\s*(?P<function_name>[\w_]+)",
        re.M | re.S,
    )

    def __init__(self, api, ffi: FFI):
        self.__api = api
        self.__ffi = ffi
        self.__lock = Lock()
        self.__counter = AtomicCounterWithCallback(self.__api.rp_shutdown)
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
        def wrapped_destructor(ptr):
            with self.__lock:
                destructor(ptr)
            self.__counter.decrement()

        @wraps(function)
        def new_function(*args):
            ret = function(*args)
            if ret != ffi.NULL:
                self.__counter.increment()
                return ffi.gc(ret, wrapped_destructor)
            else:
                return ret

        return new_function

    def close(self):
        self.__counter.mark_end()

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


ROOT = get_root()
HEADERS_DIR = ROOT / "include" / "revng" / "PipelineC"

header_paths = [
    collect_one(ROOT, ["include", "revng", "PipelineC"], "ForwardDeclarationsC.h"),
    collect_one(ROOT, ["include", "revng", "PipelineC"], "Prototypes.h"),
]

ffi = FFI()
for header_path in header_paths:
    assert header_path is not None, "Missing header file"
    with open(header_path, encoding="utf-8") as header_file:
        lines = ""
        for line in header_file:
            # TODO: use a C preprocessor to remove #include-s and macros
            if not (line.startswith("#") or line.startswith("LENGTH_HINT")):
                lines += line
            else:
                # Add newline to preserve line numbers
                lines += "\n"
        ffi.cdef(lines)


LIBRARY_PATH = collect_one(ROOT, ["lib"], "librevngPipelineC.so")
assert LIBRARY_PATH is not None, "librevngPipelineC.so not found"

ctypes_backend = CTypesBackend()

_raw_api = ffi.dlopen(str(LIBRARY_PATH), flags=ctypes_backend.RTLD_NOW | ctypes_backend.RTLD_GLOBAL)
_api = ApiWrapper(_raw_api, ffi)


def initialize(
    args: Iterable[str] = (),
    libraries: Optional[AnyPaths] = None,
    pipelines: Optional[AnyPaths] = None,
    signals_to_preserve: Iterable[int] = (),
):
    """Initialize library, must be called exactly once"""

    if libraries is None:
        libraries, _ = collect_libraries(ROOT)
    else:
        libraries = to_iterable(libraries)

    if pipelines is None:
        pipelines = collect_pipelines(ROOT)
    else:
        pipelines = to_iterable(libraries)

    path_libraries = [Path(f) for f in libraries]
    path_pipelines = [Path(f) for f in pipelines]
    assert all(p.is_file() for p in path_libraries), "Analysis libraries must all be present"
    assert all(p.is_file() for p in path_pipelines), "Pipeline files must all be present"

    args_combined = [
        "",  # Dummy value to respect argc/argv conventions
        *[f"-load={p.resolve()}" for p in path_libraries],
        *[f"-pipeline-path={p}" for p in path_pipelines],
        *args,
    ]
    _args = [ffi.new("char[]", arg.encode("utf-8")) for arg in args_combined]
    _signals_to_preserve = [int(s) for s in signals_to_preserve]

    success = _api.rp_initialize(
        len(_args),
        _args,
        len(_signals_to_preserve),
        _signals_to_preserve,
    )
    assert success, "Failed revng C API initialization"
    faulthandler.enable()


def shutdown():
    _api.close()
