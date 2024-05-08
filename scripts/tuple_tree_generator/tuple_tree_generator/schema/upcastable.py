#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

from .definition import Definition


class UpcastableDefinition(Definition):
    def __init__(self, base: Definition):
        super().__init__()
        self.base = base
