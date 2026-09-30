#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    """Identify singleton structs in models predating `IsSingleton`.

    We mark as `IsSingleton` segments, stack frames, and `struct`s marked
    `CanContainCode`, together with their parents and siblings at every
    enclosing level.

    Note: this migration might produce invalid code if a `struct` we want to
    mark as `IsSingleton` was used in multiple places.
    """

    def _referenced_struct(self, structs, type_):
        if not isinstance(type_, dict) or type_.get("Kind") != "DefinedType":
            return None
        return structs.get(type_.get("Definition", "").rsplit("/", 1)[-1])

    def migrate(self, binary):
        structs = {}
        for definition in binary.get("TypeDefinitions", []):
            if definition.get("Kind") == "StructDefinition":
                structs[f"{definition['ID']}-StructDefinition"] = definition

        parents = {}
        children = {}
        for struct in structs.values():
            children[struct["ID"]] = []
            for field in struct.get("Fields", []):
                nested = self._referenced_struct(structs, field.get("Type"))
                if nested is not None:
                    children[struct["ID"]].append(nested)
                    parents.setdefault(nested["ID"], []).append(struct)

        for segment in binary.get("Segments", []):
            struct = self._referenced_struct(structs, segment.get("Type"))
            if struct is not None:
                struct["IsSingleton"] = True

        for function in binary.get("Functions", []):
            struct = self._referenced_struct(structs, function.get("StackFrame", {}).get("Type"))
            if struct is not None:
                struct["IsSingleton"] = True

        # Code-capable structs must be singletons in the new schema. Their
        # parents and siblings at each level must also be singletons,
        # even when they cannot contain code themselves.
        worklist = [s for s in structs.values() if s.get("CanContainCode")]
        visited = set()
        while worklist:
            struct = worklist.pop()
            if struct["ID"] in visited:
                continue
            visited.add(struct["ID"])
            struct["IsSingleton"] = True

            for parent in parents.get(struct["ID"], []):
                worklist.append(parent)
                for sibling in children[parent["ID"]]:
                    sibling["IsSingleton"] = True
