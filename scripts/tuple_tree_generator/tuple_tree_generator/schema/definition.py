#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#


class Definition:
    def __init__(self, name=None):
        self.name = name
        # Names of types on which this definition depends on.
        # Not necessarily names defined by the user
        self.dependencies = set()

    def __hash__(self):
        return id(self)
