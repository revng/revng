#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    def __init__(self):
        pass

    def migrate(self, model):
        for function in model.get("Functions", {}):
            for index, comment in enumerate(function.get("Comments", [])):
                comment["Index"] = index

        model["Version"] = 4
