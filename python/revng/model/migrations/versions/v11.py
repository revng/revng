#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re

from revng.model.migrations import MigrationBase

# The x86 vector registers are renamed from `xmm` to `zmm`.
_X86_VECTOR_REGISTER = re.compile(r"^x(mm[0-7]_x86(?:_64|))$")


def _rename_register(register):
    if isinstance(register, str):
        match = _X86_VECTOR_REGISTER.match(register)
        if match is not None:
            return f"z{match.group(1)}"
    return register


class Migration(MigrationBase):
    def migrate(self, binary):
        # Register names appear in RawFunctionDefinition arguments, return
        # values and preserved registers, and in the canonical register values
        # of segments.
        for type_definition in binary.get("TypeDefinitions", []):
            if type_definition.get("Kind") != "RawFunctionDefinition":
                continue

            for register in type_definition.get("Arguments", []):
                if "Location" in register:
                    register["Location"] = _rename_register(register["Location"])

            for register in type_definition.get("ReturnValues", []):
                if "Location" in register:
                    register["Location"] = _rename_register(register["Location"])

            preserved = type_definition.get("PreservedRegisters", [])
            for index, register in enumerate(preserved):
                preserved[index] = _rename_register(register)

        for segment in binary.get("Segments", []):
            for canonical_value in segment.get("CanonicalRegisterValues", []):
                if "Register" in canonical_value:
                    canonical_value["Register"] = _rename_register(canonical_value["Register"])
