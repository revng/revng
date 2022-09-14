#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import jinja2

from .jinja_utils import loader

from ..schema import (
    Definition,
    EnumDefinition,
    ReferenceDefinition,
    ScalarDefinition,
    Schema,
    SequenceDefinition,
    StructDefinition,
    StructField,
    UpcastableDefinition,
)


environment = jinja2.Environment(
    block_start_string="<!-- ",
    block_end_string=" -->",
    variable_start_string="{{",
    variable_end_string="}}",
    comment_start_string="<!--- ",
    comment_end_string=" --->",
    loader=loader,
)


def type_name(type):
    if is_sequence(type):
        return f"list of {type_name(type.element_type)}"
    if isinstance(type, ReferenceDefinition):
        return f"reference to {type_name(type.pointee)}"
    if (isinstance(type, ScalarDefinition)
        and (type.name in ["bool", "string"] or type.name.startswith("uint"))):
        return f"`{type.name}`"
    if isinstance(type, UpcastableDefinition):
        return f"any {type_name(type.base)}"
    else:
        return f"[`{type.name}`](#{type.name})"


def indent(string, count):
    indentation = "  " * count
    return string.replace("\n\n", "\n").replace("\n", "\n" + indentation).strip()


def is_sequence(obj):
    return isinstance(obj, SequenceDefinition)


def is_scalar(obj):
    return isinstance(obj, ScalarDefinition)


def is_enum(obj):
    return isinstance(obj, EnumDefinition)


def is_struct(obj):
    return isinstance(obj, StructDefinition)


class DocsGenerator:
    def __init__(self, 
                 schema: Schema,
                 root_type,
                 string_types=None,
                 external_types=None):
        self.schema = schema
        self.root_type = root_type
        self.string_types = string_types or []
        self.external_types = external_types or []
        environment.tests.update(
            {
                "sequence": is_sequence,
                "struct": is_struct,
                "scalar": is_scalar,
                "enum": is_enum,
            }
        )
        environment.filters["indent"] = indent
        environment.filters["type_name"] = type_name
        self.template = environment.get_template("docs.md.tpl")

    def emit_docs(self) -> str:
        rendered_template = self.template.render(
            enums=self.schema.enum_definitions(),
            structs=self.schema.struct_definitions(),
            generator=self,
            schema=self.schema,
            root_type=self.root_type
        )
        return rendered_template
