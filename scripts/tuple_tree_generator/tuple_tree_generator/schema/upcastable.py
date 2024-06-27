#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from .definition import Definition


class UpcastableDefinition(Definition):
    def __init__(self, base: Definition):
        super().__init__(False)
        self.base = base
