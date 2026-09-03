#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re

from revng.model.migrations import MigrationBase

# The x86-64 vector registers are renamed `xmmN_x86_64` -> `zmmN_x86_64` to
# reflect that the model now represents the full 512-bit register. The 32-bit
# `xmmN_x86` registers are a distinct enum and must be left untouched.
_X86_64_VECTOR_REGISTER = re.compile(r"^xmm([0-7])_x86_64$")


def _rename_register(register):
    if isinstance(register, str):
        match = _X86_64_VECTOR_REGISTER.match(register)
        if match is not None:
            return f"zmm{match.group(1)}_x86_64"
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
