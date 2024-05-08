#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

from .definition import Definition


class ReferenceDefinition(Definition):
    def __init__(self, pointee: Definition, root: Definition):
        super().__init__()
        self.pointee = pointee
        self.root = root
