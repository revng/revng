#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import List


def all_strings(lst: List):
    return all(isinstance(i, str) for i in lst)
