#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from .definition import Definition


class SequenceDefinition(Definition):
    def __init__(self, sequence_type, element_type: Definition, doc=None):
        super().__init__(sequence_type == "std::vector")
        self.sequence_type = sequence_type
        self.element_type = element_type
