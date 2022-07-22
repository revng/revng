#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import black

from ..schema import Schema, SequenceStructField, StructField
from .jinja_utils import python_environment


class PythonGenerator:
    def __init__(self, schema: Schema, root_type, string_types=None, external_types=None):
        self.schema = schema
        self.root_type = root_type
        self.string_types = string_types or []
        self.external_types = external_types or []
        python_environment.filters["python_type"] = self.python_type
        python_environment.filters["python_list_type"] = self.python_list_type
        self.template = python_environment.get_template("tuple_tree_gen.py.tpl")

    def emit_python(self) -> str:
        rendered_template = self.template.render(
            enums=self.schema.enum_definitions(),
            structs=self.schema.struct_definitions(),
            generator=self,
            schema=self.schema,
        )
        formatted_rendered_template = black.format_str(
            rendered_template,
            mode=black.Mode(line_length=100),
        )
        return formatted_rendered_template

    @classmethod
    def python_type(cls, field: StructField):
        type_info = field.type_info(cls.scalar_converter)
        if type_info.root_type == "":
            if not type_info.is_sequence:
                return type_info.type
            else:
                return f"List[{type_info.type}]"
        else:
            return f"Reference[{type_info.type}, {type_info.root_type}]"

    @classmethod
    def python_list_type(cls, field: SequenceStructField):
        type_info = field.type_info(cls.scalar_converter)
        assert type_info.root_type == "", "Must not be a Reference"
        assert type_info.is_sequence, "Must be a sequence"
        return type_info.type

    @staticmethod
    def scalar_converter(type_name: str) -> str:
        if "int" in type_name:
            return "int"
        elif "string" in type_name:
            return "str"
        return type_name
