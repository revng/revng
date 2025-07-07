#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import sys
from collections.abc import Iterable as CIterable
from ctypes import CDLL
from pathlib import Path
from typing import Any, Iterable, TypeVar

from xdg import xdg_cache_home

from revng.support import SingleOrIterable

T = TypeVar("T")


def to_iterable(obj: SingleOrIterable[T]) -> Iterable[T]:
    if isinstance(obj, CIterable):
        return obj
    return (obj,)


def cache_directory() -> Path:
    if "REVNG_CACHE_DIR" in os.environ:
        return Path(os.environ["REVNG_CACHE_DIR"])
    else:
        return xdg_cache_home() / "revng"


def import_pipebox(libraries: Iterable[str | Path]) -> tuple[Any, list[Any]]:
    """Load libraries and return the pipebox module"""

    # Load the provided shared libraries, these contain static initializers
    # like RegisterPipe<T> which allows registering Pipes, Analyses and
    # Containers to be later created as python classes when the `_pipeline`
    # module is imported.
    #
    # RTLD_GLOBAL is needed due to nanobind using weak symbols to derive the TypeID
    handles = [CDLL(path, os.RTLD_NOW | os.RTLD_GLOBAL) for path in libraries]

    # Check that the module hasn't been previously imported, otherwise result
    # might be unpredictable
    assert "revng.internal._pipebox" not in sys.modules

    # Import the actual module, this will do the following:
    # 1. Create actual classes for Model, ObjectID and Kind
    # 2. Create, on the fly, classes that have been previously registered by
    #    the static initializers in the imported libraries above
    import revng.internal._pipebox as ext

    return (ext, handles)
