#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import Iterable


def all_strings(lst: Iterable) -> bool:
    return all(isinstance(i, str) for i in lst)
