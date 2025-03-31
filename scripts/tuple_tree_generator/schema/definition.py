#
# This file is distributed under the MIT License. See LICENSE.md for details.
#


from typing import Optional, Set


class Definition:
    def __init__(self, is_scalar: bool, name: Optional[str] = None):
        self.name = name
        # Names of types on which this definition depends on.
        # Not necessarily names defined by the user
        self.dependencies: Set[str] = set()
        self.is_scalar = is_scalar

    def __hash__(self):
        return id(self)
