#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    def __init__(self):
        pass

    def migrate(self, model):
        # no explicit migration necessary as both of the new data structures
        # (`model::Function::LocalVariables()` and
        # `model::Function::GotoLabels()`) are default-initializable.
        # Same for the new configuration option.

        model["Version"] = 5
