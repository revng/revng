#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from collections import Iterable as CIterable
from pathlib import Path
from typing import Iterable, TypeVar, Union

T = TypeVar("T")

SingleOrIterable = Union[T, Iterable[T]]
AnyPath = Union[str, Path]
AnyPaths = SingleOrIterable[AnyPath]


def to_iterable(obj: SingleOrIterable[T]) -> Iterable[T]:
    if isinstance(obj, CIterable):
        return obj
    return (obj,)
