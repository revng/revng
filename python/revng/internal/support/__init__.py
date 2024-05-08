#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

from collections.abc import Iterable as CIterable
from pathlib import Path
from typing import Iterable, List, TypeVar, Union

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


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    else:
        return path.read_text().strip().split("\n")
