#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from collections import defaultdict
from graphlib import TopologicalSorter
from typing import Dict, List, TypeVar

from .definition import Definition
from .enum import EnumDefinition, EnumMember
from .scalar import ScalarDefinition
from .struct import StructDefinition

T = TypeVar("T")


def assert_not_none(x: T | None) -> T:
    assert x is not None
    return x


class Schema:
    def __init__(self, raw_schema, scalar_types: List[str]):
        self._raw_schema = raw_schema

        self.version = self._raw_schema["version"]
        self.root_type = self._raw_schema["root_type"]

        # Add implicit Version field to the root type
        for definition in self._raw_schema["definitions"]:
            if definition["name"] == self.root_type:
                assert definition["type"] == "struct", "The root type must be a struct"
                definition["fields"].insert(
                    0,
                    {
                        "name": "Version",
                        "type": "uint64_t",
                        "doc": "The input's version, must match revng's schema version",
                        "optional": True,
                    },
                )

        # Fully qualified name -> object
        self.definitions: Dict[str, Definition] = self._parse_definitions()
        self.definitions["bool"] = ScalarDefinition("bool")
        self.definitions["uint64_t"] = ScalarDefinition("uint64_t")
        self.definitions["uint32_t"] = ScalarDefinition("uint32_t")
        self.definitions["uint16_t"] = ScalarDefinition("uint16_t")
        self.definitions["uint8_t"] = ScalarDefinition("uint8_t")
        self.definitions["string"] = ScalarDefinition("string")

        for scalar_type in scalar_types:
            self.definitions[scalar_type] = ScalarDefinition(scalar_type)

        self._generate_kinds()
        self._resolve_references()

    def get_definition_for(self, type_name):
        result = self.definitions.get(type_name)
        if not result:
            raise Exception(f"Unexpected type: {type_name}")
        return result

    def struct_definitions(self) -> List[StructDefinition]:
        toposorter: TopologicalSorter = TopologicalSorter()
        for struct in sorted(self.definitions.values(), key=lambda d: assert_not_none(d.name)):
            assert struct.name

            if not isinstance(struct, StructDefinition):
                continue

            toposorter.add(struct)

            for dependency in sorted(struct.dependencies):
                dep_type = self.get_definition_for(dependency)
                if isinstance(dep_type, StructDefinition):
                    toposorter.add(struct, dep_type)
                elif isinstance(dep_type, EnumDefinition) or dep_type is None:
                    pass
                else:
                    pass

        return list(toposorter.static_order())

    def enum_definitions(self) -> List[EnumDefinition]:
        return [enum for enum in self.definitions.values() if isinstance(enum, EnumDefinition)]

    def get_upcastable_types(self, base_type: StructDefinition):
        upcastable_types = set()
        for definition in self.struct_definitions():
            if definition.inherits is base_type:
                upcastable_types.add(definition)
        return sorted(upcastable_types, key=lambda t: assert_not_none(t.name))

    def _parse_definitions(self):
        definitions = {}
        for type_schema in self._raw_schema["definitions"]:
            type_name = type_schema["type"]
            if type_name == "enum":
                cls = EnumDefinition
            elif type_name == "struct":
                cls = StructDefinition
            else:
                raise ValueError(f"Invalid type name: {type_name}")

            definition = cls.from_dict(type_schema)
            definitions[definition.name] = definition

        return definitions

    def _resolve_references(self):
        inheritors_map = defaultdict(list)

        for t in self.definitions.values():
            if isinstance(t, StructDefinition):
                t.resolve_references(self)

                if t.inherits:
                    inheritors_map[t.inherits].append(t)

        for base, inheritors in inheritors_map.items():
            assert base.abstract
            base.inheritors = inheritors

    def _generate_kinds(self):
        children = defaultdict(list)
        for _, value in self.definitions.items():
            if isinstance(value, StructDefinition) and value._inherits is not None:
                parent = self.get_definition_for(value._inherits)
                children[parent].append(value)

        for parent, child_list in children.items():
            kind_field = [f for f in parent.fields if f.name == "Kind"]
            if len(kind_field) == 0:
                raise ValueError("Kind field must be present in abstract classes")
            if kind_field[0].type != f"{parent.name}Kind":
                raise ValueError(
                    f"Kind field in {parent.name} must have the type "
                    + f"{parent.name}Kind'. "
                    + "This enum will be autogenerated with the child names "
                    + "of this class."
                )
            parent.children = child_list

            name = f"{parent.name}Kind"
            kind_enum = EnumDefinition(
                name=name,
                doc=f"The list of child types `{parent.name}` can be upcasted to",
                members=[EnumMember(name=c.name) for c in child_list],
            )
            kind_enum.autogenerated = True
            self.definitions[name] = kind_enum


def remove_prefix(s: str, prefix: str):
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


def remove_suffix(s: str, suffix: str):
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s
