#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import graphlib
from typing import Dict, List

from .definition import Definition
from .enum import EnumDefinition
from .struct import StructDefinition


class Schema:
    def __init__(self, raw_schema, base_namespace: str):
        self._raw_schema = raw_schema

        self.base_namespace = base_namespace
        self.generated_namespace = f"{base_namespace}::generated"

        # fully qualified name -> object
        self.definitions: Dict[str, Definition] = self._parse_definitions()
        self._resolve_references()

    def get_definition_for(self, type_name):
        if self.definitions.get(type_name):
            return self.definitions.get(type_name)

        stripped_type_name = remove_prefix(type_name, f"{self.generated_namespace}::")
        stripped_type_name = remove_prefix(stripped_type_name, f"{self.base_namespace}::")

        if self.definitions.get(stripped_type_name):
            return self.definitions.get(stripped_type_name)

        elif self.definitions.get(f"{self.base_namespace}::{stripped_type_name}"):
            return self.definitions.get(f"{self.base_namespace}::{stripped_type_name}")

        elif self.definitions.get(f"{self.generated_namespace}::{stripped_type_name}"):
            return self.definitions.get(f"{self.generated_namespace}::{stripped_type_name}")

        return None

    def struct_definitions(self) -> List[StructDefinition]:
        toposorter = graphlib.TopologicalSorter()
        for struct in self.definitions.values():
            if not isinstance(struct, StructDefinition):
                continue

            toposorter.add(struct)

            for dependency in struct.dependencies:
                dep_type = self.get_definition_for(dependency)
                if isinstance(dep_type, StructDefinition):
                    toposorter.add(struct, dep_type)
                elif isinstance(dep_type, EnumDefinition):
                    pass
                elif dep_type is None:
                    pass
                else:
                    breakpoint()

        return list(toposorter.static_order())

    def enum_definitions(self) -> List[EnumDefinition]:
        return [enum for enum in self.definitions.values() if isinstance(enum, EnumDefinition)]

    def get_upcastable_types(self, base_type: StructDefinition):
        upcastable_types = set()
        for definition in self.struct_definitions():
            if definition.inherits is base_type:
                upcastable_types.add(definition)
        return upcastable_types

    def _parse_definitions(self):
        definitions = {}
        for type_schema in self._raw_schema:
            type_name = type_schema["type"]
            if type_name == "enum":
                cls = EnumDefinition
            elif type_name == "struct":
                cls = StructDefinition
            else:
                raise ValueError(f"Invalid type name: {type_name}")

            definition = cls.from_dict(type_schema, self.base_namespace)
            definitions[definition.fullname] = definition

        return definitions

    def _resolve_references(self):
        for t in self.definitions.values():
            t.resolve_references(self)


def remove_prefix(s: str, prefix: str):
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s
