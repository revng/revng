#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    def __init__(self):
        pass

    def migrate(self, model):
        disassembly_config = model.get("Configuration", {}).get("Disassembly", {})
        if "UseATTSyntax" in disassembly_config:
            disassembly_config["UseX86ATTSyntax"] = disassembly_config["UseATTSyntax"]
            del disassembly_config["UseATTSyntax"]
        model["Version"] = 3
