#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase

# The ABIs of the 32-bit ARM and MIPS architectures are split into a
# hard-float and a soft-float variant. The undivided ones are renamed after
# the hard-float variant, which is what they used to describe.
_RENAMED_ABIS = {
    "AAPCS": "AAPCS_hardfloat",
    "SystemV_MIPS_o32": "SystemV_MIPS_o32_hardfloat",
    "SystemV_MIPSEL_o32": "SystemV_MIPSEL_o32_hardfloat",
}


class Migration(MigrationBase):
    """Rename the ARM and MIPS ABIs to their hard-float variants."""

    def migrate(self, binary):
        # The ABI appears in three places: the `DefaultABI` and `TargetABI`
        # of the binary itself, and the `ABI` of every C ABI function
        # definition.
        for field in ("DefaultABI", "TargetABI"):
            self._rename(binary, field)

        for definition in binary.get("TypeDefinitions", []):
            if definition.get("Kind") == "CABIFunctionDefinition":
                self._rename(definition, "ABI")

    def _rename(self, dictionary, field):
        abi = dictionary.get(field)
        if abi in _RENAMED_ABIS:
            dictionary[field] = _RENAMED_ABIS[abi]
