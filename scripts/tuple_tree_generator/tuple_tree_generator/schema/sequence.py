#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

from .definition import Definition


class SequenceDefinition(Definition):
    def __init__(self, sequence_type, element_type: Definition, doc=None):
        super().__init__()
        self.sequence_type = sequence_type
        self.element_type = element_type
