#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    def _update_attributes(self, attributes):
        for index, value in enumerate(attributes):
            if value == "Inline":
                attributes[index] = "AlwaysInline"

    def migrate(self, binary):
        for function in binary.get("Functions", {}):
            self._update_attributes(function.get("Attributes", []))

            for callsite in function.get("CallSitePrototypes", []):
                self._update_attributes(callsite.get("Attributes", []))

        for function in binary.get("ImportedDynamicFunctions", {}):
            self._update_attributes(function.get("Attributes", []))
