#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    def __init__(self):
        pass

    def migrate(self, model):
        model["Version"] = 2
