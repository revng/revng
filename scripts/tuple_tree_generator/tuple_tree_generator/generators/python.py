#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import black
from jinja2 import Environment

from tuple_tree_generator.schema import Definition, EnumDefinition, ReferenceDefinition
from tuple_tree_generator.schema import ScalarDefinition, Schema, SequenceDefinition
from tuple_tree_generator.schema import StructDefinition, StructField, UpcastableDefinition
from tuple_tree_generator.schema.struct import ReferenceStructField, SequenceStructField
from tuple_tree_generator.schema.struct import SimpleStructField

from .jinja_utils import int_re, is_reference_struct_field, is_sequence_struct_field
from .jinja_utils import is_simple_struct_field, loader


class PythonGenerator:
    def __init__(self, schema: Schema, root_type, string_types=None, external_types=None):
        self.schema = schema
        self.root_type = root_type
        self.string_types = string_types or []
        self.external_types = external_types or []
        self.jinja_environment = Environment(
            block_start_string="##",
            block_end_string="##",
            variable_start_string="'",
            variable_end_string="'",
            comment_start_string="###",
            comment_end_string="###",
            loader=loader,
        )
        self.jinja_environment.filters["python_type"] = self.python_type
        self.jinja_environment.filters["docstring"] = self.render_docstring
        self.jinja_environment.filters["default_value"] = self.default_value
        self.jinja_environment.filters["gen_key"] = self.gen_key
        self.jinja_environment.filters["key_parser"] = self.key_parser
        self.jinja_environment.tests["simple_field"] = is_simple_struct_field
        self.jinja_environment.tests["sequence_field"] = is_sequence_struct_field
        self.jinja_environment.tests["reference_field"] = is_reference_struct_field
        self.template = self.jinja_environment.get_template("tuple_tree_gen.py.tpl")

    def emit_python(self) -> str:
        rendered_template = self.template.render(
            enums=self.schema.enum_definitions(),
            structs=self.schema.struct_definitions(),
            generator=self,
            schema=self.schema,
            root_type=self.schema.root_type,
            version=self.schema.version,
        )
        formatted_rendered_template = black.format_str(
            rendered_template,
            mode=black.Mode(line_length=100),
        )
        return formatted_rendered_template

    @classmethod
    def python_type(cls, field: StructField):
        resolved_type = field.resolved_type
        return cls._python_type(resolved_type)

    @classmethod
    def _python_type(cls, resolved_type: Definition):
        assert isinstance(resolved_type, Definition)
        if isinstance(resolved_type, StructDefinition):
            return resolved_type.name
        elif isinstance(resolved_type, SequenceDefinition):
            return f"List[{cls._python_type(resolved_type.element_type)}]"
        elif isinstance(resolved_type, EnumDefinition):
            return resolved_type.name
        elif isinstance(resolved_type, ScalarDefinition):
            assert resolved_type.name
            if resolved_type.name == "string":
                return "str"
            elif int_re.match(resolved_type.name):
                return "int"
            else:
                return resolved_type.name
        elif isinstance(resolved_type, ReferenceDefinition):
            pointee = cls._python_type(resolved_type.pointee)
            root = cls._python_type(resolved_type.root)
            return f"Reference[{pointee}, {root}]"
        elif isinstance(resolved_type, UpcastableDefinition):
            return cls._python_type(resolved_type.base)
        else:
            assert False

    @staticmethod
    def gen_key(struct: StructDefinition) -> str:
        fields = [f"{{self.{f.name}}}" for f in struct.key_fields]
        return f"f\"{'-'.join(fields)}\""

    @staticmethod
    def key_parser(struct: StructDefinition) -> str:
        fields = [f'"{f.name}": parts[{i}]' for i, f in enumerate(struct.key_fields)]
        return ",".join(fields)

    @staticmethod
    def render_docstring(docstr: str, indent=1):
        if not docstr:
            return ""
        # Replace '\' with '\\' to avoid "SyntaxWarning: invalid escape sequence"
        lines = docstr.replace("\\", "\\\\").splitlines(keepends=False)
        rendered_docstring = '"""'
        rendered_docstring += ("\n" + "    " * indent).join(lines)
        if len(lines) > 1:
            rendered_docstring += "\n    " * indent
        rendered_docstring += '"""'
        return rendered_docstring

    def default_value(self, field: StructField):
        if isinstance(field, ReferenceStructField):
            return '""'
        elif isinstance(field, SequenceStructField):
            return "[]"
        elif isinstance(field, SimpleStructField):
            if field.type == "string" or field.type in self.string_types:
                return '""'
            elif field.type == "bool":
                return "False"
            elif int_re.match(field.type):
                return "0"
            elif field.upcastable:
                return "None"
            else:
                return f"{field.type}()"
        else:
            raise ValueError()
