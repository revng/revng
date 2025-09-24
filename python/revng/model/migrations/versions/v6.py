#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    def __init__(self):
        pass

    def migrate(self, model):
        naming_config = model.get("Configuration", {}).get("Naming", {})
        if "UndefinedValuePrefix" in naming_config:
            del naming_config["UndefinedValuePrefix"]
        model["Version"] = 6
