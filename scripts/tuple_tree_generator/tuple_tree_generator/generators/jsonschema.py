#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
import yaml
from typing import Dict

from .generator import Generator
from ..definition import EnumDefinition
from ..definition import StructDefinition, StructField, SimpleStructField, SequenceStructField, ReferenceStructField


class JSONSchemaGenerator(Generator):
    def __init__(self, schema, base_namespace: str, root_type):
        super().__init__(schema, base_namespace)
        self.root_type = root_type

    def emit(self) -> Dict[str, str]:
        sources = {
            "jsonschema.yml": self._emit_jsonschema(self.root_type)
        }
        return sources

    def _emit_jsonschema(self, root_type: str):
        jsonschema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$ref": f"#/definitions/{root_type}",
            "definitions": {
                "Typeref": {
                    "type": "string",
                },
                "MetaAddress": {
                    "type": "string",
                },
            },
        }
        definitions = jsonschema["definitions"]

        for type_definition in self._definitions.values():
            assert type_definition.name not in definitions
            if isinstance(type_definition, EnumDefinition):
                definition = self._enum_jsonschema(type_definition)
            elif isinstance(type_definition, StructDefinition):
                definition = self._struct_jsonschema(type_definition)
            else:
                raise ValueError()

            definitions[type_definition.tag] = definition

        return yaml.dump(jsonschema)

    def _enum_jsonschema(self, definition: EnumDefinition):
        members = ["Invalid"]
        members += [m.name for m in definition.members]

        jsonschema = {
            "enum": members,
            "title": definition.tag,
        }
        if definition.doc:
            jsonschema["description"] = definition.doc

        return jsonschema

    def _struct_jsonschema(self, definition: StructDefinition):
        jsonschema = {
            "type": "object",
            "title": definition.tag,
            "additionalProperties": False,
        }
        if definition.doc:
            jsonschema["description"] = definition.doc

        # TODO: a type can be upcastable even if it is not abstract
        if definition.abstract:
            upcastable_to = self._get_upcastable_types(definition)
            jsonschema["oneOf"] = [
                {"$ref": f"#/definitions/{f.tag}"} for f in upcastable_to
            ]
        else:
            properties = {}

            for field in definition.all_fields:
                field_definition = self._convert_struct_field(field)
                properties[field.name] = field_definition

            jsonschema["required"] = [f.name for f in definition.all_required_fields]
            jsonschema["properties"] = properties

        return jsonschema

    def _convert_struct_field(self, field: StructField):
        # TODO: don't hardcode non-standard types
        revng_string_types = [
            "Identifier",
        ]

        if isinstance(field, SimpleStructField):
            internal_type = self.get_definition_for(field.type)
            int_re = re.compile(r"(u)?int(8|16|32|64)_t")
            if internal_type:
                schema = {
                    "$ref": f"#/definitions/{internal_type.name}"
                }
            elif field.type == "MetaAddress":
                schema = {
                    "$ref": "#/definitions/MetaAddress"
                }
            else:
                if field.type in ["std::string", "string", *revng_string_types]:
                    field_type = "string"
                elif field.type == "bool":
                    field_type = "boolean"
                elif int_re.fullmatch(field.type):
                    field_type = "integer"
                else:
                    raise ValueError()
                schema = {"type": field_type}

        elif isinstance(field, SequenceStructField):
            internal_type = self.get_definition_for(field.element_type)
            if internal_type:
                schema = {
                    "type": "array",
                    "items": {
                        "$ref": f"#/definitions/{internal_type.name}"
                    }
                }
            elif field.element_type in ["std::string", "string", *revng_string_types]:
                schema = {
                    "type": "array",
                    "items": {
                        "type": "string",
                    }
                }
            else:
                raise ValueError()
        elif isinstance(field, ReferenceStructField):
            schema = {"$ref": "#/definitions/Typeref"}
        else:
            raise ValueError()

        return schema
