#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from .definition import Definition


class ScalarDefinition(Definition):
    def __init__(self, name):
        super().__init__(True, name)
