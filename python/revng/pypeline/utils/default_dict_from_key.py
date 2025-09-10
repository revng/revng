#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from collections.abc import Mapping
from typing import Callable, Dict, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class DefaultDictFromKey(Generic[K, V], Mapping[K, V]):
    """
    A dictionary that, when queried with a key not currently present in the dictionary,
    automatically creates it initializing it using a custom function of the key.
    """

    def __init__(self, factory: Callable[[K], V]):
        self._dictionary: Dict[K, V] = {}
        self._factory = factory

    def __getitem__(self, key: K) -> V:
        if key not in self._dictionary:
            self._dictionary[key] = self._factory(key)
        return self._dictionary[key]

    def __iter__(self):
        return self._dictionary.__iter__()

    def __len__(self):
        return self._dictionary.__len__()
