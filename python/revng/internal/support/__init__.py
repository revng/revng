#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from collections.abc import Iterable as CIterable
from pathlib import Path
from typing import Iterable, TypeVar

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
