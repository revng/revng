#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase

# The type and the size corresponding to each of the old relocation types
_RELOCATION_TYPES = {
    "WriteAbsoluteAddress32": ("Absolute", 4),
    "WriteAbsoluteAddress64": ("Absolute", 8),
    "AddAbsoluteAddress32": ("Absolute", 4),
    "AddAbsoluteAddress64": ("Absolute", 8),
    "WriteRelativeAddress32": ("PCRelative", 4),
    "WriteRelativeAddress64": ("PCRelative", 8),
    "AddRelativeAddress32": ("PCRelative", 4),
    "AddRelativeAddress64": ("PCRelative", 8),
}


def _make_primitive_sizes_explicit(node):
    if isinstance(node, dict):
        if node.get("Kind") == "PrimitiveType":
            node.setdefault("Size", 0)
        for value in node.values():
            _make_primitive_sizes_explicit(value)
    elif isinstance(node, list):
        for value in node:
            _make_primitive_sizes_explicit(value)


def _migrate_relocations(relocations):
    # TODO: `Add*` relocations also add the value in place, which can only be
    #       read from the binary: a proper migration needs access to it.
    # TODO: migrations have no way to report errors, such as the information
    #       lost here.
    migrated = []
    keys = set()
    for relocation in relocations:
        if relocation.get("Type") in _RELOCATION_TYPES:
            relocation["Type"], relocation["Size"] = _RELOCATION_TYPES[relocation["Type"]]

        # Merging `Add*` into `Write*` might produce duplicate keys
        key = (relocation.get("Address"), relocation.get("Type"))
        if key not in keys:
            keys.add(key)
            migrated.append(relocation)

    relocations[:] = migrated


class Migration(MigrationBase):
    """Make primitive sizes explicit, address space 0 position dependent and
    move the size of relocations out of their type.

    Absent fields take the default value declared in the schema, which, for
    `PrimitiveType.Size`, is -1. In previous versions, an absent `Size`
    (e.g., in `void`) stood for 0.

    Address spaces are position independent by default, which makes analyses
    rely on relocations alone to find pointers in global data. Models
    predating `AddressSpaces` might lack such relocations, so we keep looking
    for pointers in the raw content of their segments.

    Relocation types no longer encode the size, which is now in
    `Relocation.Size`. `Add*` types, which add to the value in place, are
    merged into the corresponding `Write*` ones, losing that value.
    """

    def migrate(self, binary):
        _make_primitive_sizes_explicit(binary)

        for segment in binary.get("Segments", []):
            _migrate_relocations(segment.get("Relocations", []))
        for function in binary.get("ImportedDynamicFunctions", []):
            _migrate_relocations(function.get("Relocations", []))

        address_spaces = binary.setdefault("AddressSpaces", [])
        for address_space in address_spaces:
            if address_space.get("Index", 0) == 0:
                return
        address_spaces.insert(0, {"Index": 0, "IsPositionIndependent": False})
