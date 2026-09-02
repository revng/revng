#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import sys
import tempfile
from collections.abc import Iterable as CIterable
from ctypes import CDLL
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, TypeVar

import yaml
from xdg import xdg_cache_home, xdg_config_home

from revng.support import SingleOrIterable

T = TypeVar("T")


def to_iterable(obj: SingleOrIterable[T]) -> Iterable[T]:
    if isinstance(obj, CIterable):
        return obj
    return (obj,)


def merge(a, b, path=()):
    """
    Merge two JSON objects recursively.
    Lists are concatenated.
    Dictionaries are merged.
    If a dictionary has conclifting scalar values, it fails.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        result = {**a}
        for k, v in b.items():
            if k in result:
                result[k] = merge(result[k], v, path + (k,))
            else:
                result[k] = v
        return result
    if isinstance(a, list) and isinstance(b, list):
        return (
            [merge(x, y, path + (i,)) for i, (x, y) in enumerate(zip(a, b))]
            + a[len(b) :]
            + b[len(a) :]
        )
    if a == b:
        return a
    p = "/".join(str(x) for x in path) or "<root>"
    raise ValueError(f"conflicting values at {p}: {a!r} vs {b!r}")


@lru_cache
def configuration():
    result = {}

    paths = [
        Path("/etc/revng/configuration.yml"),
        xdg_config_home() / "revng" / "configuration.yml",
    ]

    for path in paths:
        if path.exists():
            with open(path, "r") as file:
                result = merge(result, yaml.safe_load(file))

    return result


_cache_directory_fallback: tempfile.TemporaryDirectory | None = None


def cache_directory() -> Path:
    global _cache_directory_fallback
    if "cache-path" in configuration():
        return Path("cache-path")
    candidate = xdg_cache_home() / "revng"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except OSError:
        # If we can't create a directory, typically for permission reasons, use
        # a temporary directory
        if _cache_directory_fallback is None:
            _cache_directory_fallback = tempfile.TemporaryDirectory(prefix="revng-cache-")
        return Path(_cache_directory_fallback.name)


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
