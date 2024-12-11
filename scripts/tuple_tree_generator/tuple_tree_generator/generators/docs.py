#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from collections import defaultdict

import jinja2

from ..schema import EnumDefinition, ReferenceDefinition, ScalarDefinition, Schema
from ..schema import SequenceDefinition, StructDefinition, UpcastableDefinition
from .jinja_utils import loader

environment = jinja2.Environment(
    block_start_string="<!-- ",
    block_end_string=" -->",
    variable_start_string="{{",
    variable_end_string="}}",
    comment_start_string="<!--- ",
    comment_end_string=" --->",
    loader=loader,
)


def type_name(target_type):
    if is_sequence(target_type):
        return f"list of {type_name(target_type.element_type)}"
    if isinstance(target_type, ReferenceDefinition):
        return f"reference to {type_name(target_type.pointee)}"
    if isinstance(target_type, ScalarDefinition):
        return f"`{target_type.name}`"
    if isinstance(target_type, UpcastableDefinition):
        return f"any {type_name(target_type.base)}"
    else:
        return f"[`{target_type.name}`](#{target_type.name})"


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


def is_upcastable(obj):
    return isinstance(obj, UpcastableDefinition)


def leaf(target_type):
    if is_sequence(target_type):
        return target_type.element_type

    if isinstance(target_type, ReferenceDefinition):
        return target_type.pointee

    if isinstance(target_type, UpcastableDefinition):
        return target_type.base

    return target_type


class DocsGenerator:
    def __init__(self, schema: Schema, root_type):
        self.schema = schema
        self.root_type = root_type
        environment.tests.update(
            {
                "sequence": is_sequence,
                "struct": is_struct,
                "scalar": is_scalar,
                "enum": is_enum,
                "upcastable": is_upcastable,
            }
        )
        environment.filters["indent"] = indent
        environment.filters["type_name"] = type_name
        self.template = environment.get_template("docs.md.tpl")

    def emit_docs(self) -> str:
        references = defaultdict(set)

        enums = self.schema.enum_definitions()
        structs = self.schema.struct_definitions()
        for struct in structs:
            for field in struct.fields:
                references[leaf(field.resolved_type).name].add((struct, field))

        rendered_template = self.template.render(
            enums=enums,
            structs=structs,
            generator=self,
            schema=self.schema,
            root_type=self.root_type,
            references_map=dict(references),
        )
        return rendered_template
