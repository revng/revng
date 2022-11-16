#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import black
import jinja2

from ..schema import Definition, EnumDefinition, ReferenceDefinition, ScalarDefinition, Schema
from ..schema import SequenceDefinition, StructDefinition, StructField, UpcastableDefinition, ReferenceStructField,    SequenceStructField,    SimpleStructField
from .jinja_utils import loader

python_environment = jinja2.Environment(
    block_start_string="##",
    block_end_string="##",
    variable_start_string="'",
    variable_end_string="'",
    comment_start_string="###",
    comment_end_string="###",
    loader=loader,
)


def render_python_docstring(docstr: str, indent=1):
    if not docstr:
        return ""
    lines = docstr.splitlines(keepends=False)
    rendered_docstring = '"""'
    rendered_docstring += ("\n" + "    " * indent).join(lines)
    if len(lines) > 1:
        rendered_docstring += "\n    " * indent
    rendered_docstring += '"""'
    return rendered_docstring


python_environment.filters["docstring"] = render_python_docstring


def is_simple_struct_field(obj):
    return isinstance(obj, SimpleStructField)


def is_sequence_struct_field(obj):
    return isinstance(obj, SequenceStructField)


def is_reference_struct_field(obj):
    return isinstance(obj, ReferenceStructField)


python_environment.tests.update(
    {
        "simple_field": is_simple_struct_field,
        "sequence_field": is_sequence_struct_field,
        "reference_field": is_reference_struct_field,
    }
)


class PythonGenerator:
    def __init__(self, schema: Schema, root_type, string_types=None, external_types=None):
        self.schema = schema
        self.root_type = root_type
        self.string_types = string_types or []
        self.external_types = external_types or []
        python_environment.filters["python_type"] = self.python_type
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
            if resolved_type.name == "string":
                return "str"
            elif "int" in resolved_type.name:
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
