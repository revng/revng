#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Any, Dict, MutableMapping

import yaml

from ..schema import EnumDefinition, ReferenceStructField, ScalarDefinition, Schema
from ..schema import SequenceStructField, SimpleStructField, StructDefinition, StructField
from .jinja_utils import int_re


class JSONSchemaGenerator:
    def __init__(self, schema: Schema, root_type, string_types=None, separate_string_types=None):
        self.schema = schema
        self.root_type = root_type
        self.string_types = string_types or []
        self.separate_string_types = separate_string_types or []

    def emit_jsonschema(self) -> str:
        """Returns the JSON schema as a string"""
        jsonschema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$ref": f"#/definitions/{self.root_type}",
            "definitions": {
                "Reference": {
                    "type": "string",
                },
            },
        }
        definitions: Dict[str, Any] = jsonschema["definitions"]  # type: ignore

        for t in self.separate_string_types:
            definitions[t] = {"type": "string"}

        for type_definition in self.schema.definitions.values():
            if isinstance(type_definition, EnumDefinition):
                definition = self._enum_jsonschema(type_definition)
            elif isinstance(type_definition, StructDefinition):
                definition = self._struct_jsonschema(type_definition)
            elif isinstance(type_definition, ScalarDefinition):
                definition = None
            else:
                raise ValueError()

            if definition is not None:
                assert type_definition.name not in definitions
                assert type_definition.name
                definitions[type_definition.name] = definition

        return yaml.dump(jsonschema)

    @staticmethod
    def _enum_jsonschema(definition: EnumDefinition):
        members = ["Invalid"]
        members += [m.name for m in definition.members]

        jsonschema = {
            "enum": members,
            "title": definition.name,
        }
        if definition.doc:
            jsonschema["description"] = definition.doc

        return jsonschema

    def _struct_jsonschema(self, definition: StructDefinition):
        jsonschema: MutableMapping[str, Any] = {
            "type": "object",
            "title": definition.name,
            "additionalProperties": False,
        }
        if definition.doc:
            jsonschema["description"] = definition.doc

        # TODO: a type can be upcastable even if it is not abstract
        if definition.abstract:
            upcastable_to = self.schema.get_upcastable_types(definition)
            jsonschema["oneOf"] = [{"$ref": f"#/definitions/{f.name}"} for f in upcastable_to]
        else:
            properties = {}

            for field in definition.all_fields:
                field_definition = self._convert_struct_field(field)
                properties[field.name] = field_definition

            jsonschema["required"] = [f.name for f in definition.all_required_fields]
            jsonschema["properties"] = properties

        return jsonschema

    def _convert_struct_field(self, field: StructField):
        if isinstance(field, SimpleStructField):
            internal_type = self.schema.get_definition_for(field.type)
            schema = self._get_schema_for_scalar(field.type, internal_type)

        elif isinstance(field, SequenceStructField):
            internal_element_type = self.schema.get_definition_for(field.element_type)
            element_schema = self._get_schema_for_scalar(field.element_type, internal_element_type)
            schema = {"type": "array", "items": element_schema}
        elif isinstance(field, ReferenceStructField):
            schema = {"$ref": "#/definitions/Reference"}
        else:
            raise ValueError()

        return schema

    def _get_schema_for_scalar(self, type_name, internal_type):
        if internal_type:
            schema = {"$ref": f"#/definitions/{internal_type.name}"}
        elif type_name in self.separate_string_types:
            schema = {"$ref": f"#/definitions/{type_name}"}
        else:
            if type_name in ["string", *self.string_types]:
                field_type = "string"
            elif type_name == "bool":
                field_type = "boolean"
            elif int_re.fullmatch(type_name):
                field_type = "integer"
            else:
                raise ValueError()
            schema = {"type": field_type}
        return schema
