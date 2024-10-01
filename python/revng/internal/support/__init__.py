#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from collections.abc import Iterable as CIterable
from pathlib import Path
from typing import Iterable, List, TypeVar, Union

from xdg import xdg_cache_home

T = TypeVar("T")

SingleOrIterable = Union[T, Iterable[T]]
AnyPath = Union[str, Path]
AnyPaths = SingleOrIterable[AnyPath]


def to_iterable(obj: SingleOrIterable[T]) -> Iterable[T]:
    if isinstance(obj, CIterable):
        return obj
    return (obj,)


# We assume this file is in <root>/lib/python$VERSION/site-packages/revng/support
def get_root() -> Path:
    return (Path(__file__) / "../../../../../../../").resolve()


def cache_directory() -> Path:
    if "REVNG_CACHE_DIR" in os.environ:
        return Path(os.environ["REVNG_CACHE_DIR"])
    else:
        return xdg_cache_home() / "revng"


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    else:
        return path.read_text().strip().split("\n")
