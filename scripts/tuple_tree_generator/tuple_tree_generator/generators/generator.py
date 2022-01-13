#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Iterator

from ..definition import Definition, EnumDefinition, StructDefinition


class Generator(ABC):
    def __init__(self, schema, base_namespace: str):
        self.schema = schema
        self.base_namespace = base_namespace
        self.generated_namespace = f"{base_namespace}::generated"

        self._type_to_class = {
            "enum": EnumDefinition,
            "struct": StructDefinition,
        }
        # fully qualified name -> object
        self._definitions: Dict[str, Definition] = {}

        self._parse_definitions()
        self._resolve_references()

        self.depgraph = self._build_depgraph()
        self._inverse_depgraph = self._build_inverse_depgraph()

    def _build_depgraph(self):
        depgraph = {}
        for definition in self._definitions.values():
            depgraph[definition.fullname] = set()
            for dependency_name in definition.dependencies:
                dependency_type = self.get_definition_for(dependency_name)
                if dependency_type:
                    depgraph[definition.fullname].add(dependency_type.fullname)
        return depgraph

    def _build_inverse_depgraph(self):
        inverse_depgraph = defaultdict(set)
        for definition in self._definitions.values():
            for dependency in definition.dependencies:
                inverse_depgraph[dependency].add(definition.fullname)
        return inverse_depgraph

    def _parse_definitions(self):
        for type_schema in self.schema:
            type_name = type_schema["type"]
            definition = self._type_to_class[type_name].from_dict(type_schema, self.base_namespace)
            self._definitions[definition.fullname] = definition

    def _resolve_references(self):
        for type in self._definitions.values():
            type.resolve_references(self)

    def get_definition_for(self, type_name):
        if self._definitions.get(type_name):
            return self._definitions.get(type_name)

        stripped_type_name = remove_prefix(type_name, f"{self.generated_namespace}::")
        stripped_type_name = remove_prefix(stripped_type_name, f"{self.base_namespace}::")

        if self._definitions.get(stripped_type_name):
            return self._definitions.get(stripped_type_name)

        elif self._definitions.get(f"{self.base_namespace}::{stripped_type_name}"):
            return self._definitions.get(f"{self.base_namespace}::{type_name}")

        elif self._definitions.get(f"{self.generated_namespace}::{stripped_type_name}"):
            return self._definitions.get(f"{self.generated_namespace}::{type_name}")

        return None

    @abstractmethod
    def emit(self) -> Dict[str, str]:
        """Returns a dict filename -> content"""
        raise NotImplementedError()

    def struct_definitions(self) -> Iterator[StructDefinition]:
        for definition in self._definitions.values():
            if isinstance(definition, StructDefinition):
                yield definition

    def enum_definitions(self) -> Iterator[EnumDefinition]:
        for definition in self._definitions.values():
            if isinstance(definition, EnumDefinition):
                yield definition

    def _get_upcastable_types(self, base_type: StructDefinition):
        upcastable_types = set()
        for definition in self.struct_definitions():
            if definition.inherits is base_type:
                upcastable_types.add(definition)
        return upcastable_types


def remove_prefix(s: str, prefix: str):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s
