#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import black
from jinja2 import Environment

from tuple_tree_generator.schema import Definition, EnumDefinition, ReferenceDefinition
from tuple_tree_generator.schema import ScalarDefinition, Schema, SequenceDefinition
from tuple_tree_generator.schema import StructDefinition, StructField, UpcastableDefinition
from tuple_tree_generator.schema.struct import SimpleStructField

from .jinja_utils import int_re, is_reference_struct_field, is_sequence_struct_field
from .jinja_utils import is_simple_struct_field, loader


class PythonGenerator:
    def __init__(
        self,
        schema: Schema,
        output,
        string_types=[],
        external_types=[],
        mixins: list[str] = [],
    ):
        self.schema = schema
        self.root_type = schema.root_type
        self.output = output
        self.string_types = string_types
        self.external_types = external_types
        self.mixins = set(mixins)
        self.jinja_environment = Environment(
            block_start_string="##",
            block_end_string="##",
            variable_start_string="#{",
            variable_end_string="}#",
            comment_start_string="###",
            comment_end_string="###",
            loader=loader,
        )
        self.jinja_environment.filters["python_type"] = self.python_type
        self.jinja_environment.filters["type_metadata"] = self.type_metadata
        self.jinja_environment.filters["docstring"] = self.render_docstring
        self.jinja_environment.filters["get_default_value"] = self.get_default_value
        self.jinja_environment.filters["gen_key"] = self.gen_key
        self.jinja_environment.filters["key_parser"] = self.key_parser
        self.jinja_environment.filters["get_mixins"] = self.get_mixins
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
        return black.format_str(
            rendered_template,
            mode=black.Mode(line_length=100),
        )

    @classmethod
    def python_type(cls, field: StructField | str):
        if isinstance(field, str):
            if field == "string":
                return "str"
            if int_re.match(field):
                return "int"
            else:
                return field
        resolved_type = field.resolved_type
        assert resolved_type
        return cls._python_type(resolved_type)

    @classmethod
    def _python_type(cls, resolved_type: Definition):
        assert isinstance(resolved_type, Definition)
        if isinstance(resolved_type, StructDefinition):
            return resolved_type.name
        if isinstance(resolved_type, SequenceDefinition):
            return f"TypedList[{cls._python_type(resolved_type.element_type)}]"
        if isinstance(resolved_type, EnumDefinition):
            return resolved_type.name
        if isinstance(resolved_type, ScalarDefinition):
            assert resolved_type.name
            if resolved_type.name == "string":
                return "str"
            if int_re.match(resolved_type.name):
                return "int"
            return resolved_type.name
        if isinstance(resolved_type, ReferenceDefinition):
            pointee = cls._python_type(resolved_type.pointee)
            root = cls._python_type(resolved_type.root)
            return f"Reference[{pointee}, {root}]"
        if isinstance(resolved_type, UpcastableDefinition):
            return cls._python_type(resolved_type.base)
        assert False

    def scalar_converter(self, type_name: str) -> str:
        if type_name == "string":
            return "str"
        if type_name == "bool":
            return "bool"
        if int_re.fullmatch(type_name):
            return "int"
        if type_name in self.external_types + self.string_types:
            return type_name
        raise Exception(f"Unexpected scalar: {type_name}")

    def _real_type(self, field: StructField) -> str:
        resolved_type = field.resolved_type

        if isinstance(resolved_type, SequenceDefinition):
            resolved_type = resolved_type.element_type

        if isinstance(resolved_type, UpcastableDefinition):
            resolved_type = resolved_type.base

        assert resolved_type
        real_type = resolved_type.name

        if isinstance(resolved_type, ScalarDefinition):
            return self.scalar_converter(real_type)

        return real_type

    def type_metadata(self, field: StructField) -> str:
        real_type = self._real_type(field)
        possible_values = None
        external = False
        if real_type in {"str", "bool", "int"}:
            hint_type = real_type
            ctor = "native"
        elif real_type in [e.name for e in self.schema.enum_definitions()]:
            hint_type = "str"
            ctor = "enum"
            possible_values = f"{real_type}"
        elif isinstance(field.resolved_type, ReferenceDefinition):
            hint_type = "Reference"
            ctor = "parse"
        elif real_type in [
            *(s.name for s in self.schema.struct_definitions() if s.abstract),
            *(self.string_types),
        ]:
            hint_type = real_type
            ctor = "parse"
            if real_type in self.string_types:
                external = True
        else:
            hint_type = real_type
            ctor = "class"
            if real_type in self.external_types:
                external = True

        assert hint_type

        is_sequence = isinstance(field.resolved_type, SequenceDefinition)
        if isinstance(field.resolved_type, SequenceDefinition):
            is_abstract = isinstance(field.resolved_type.element_type, UpcastableDefinition)
        else:
            is_abstract = isinstance(field.resolved_type, UpcastableDefinition)

        return (
            f'{{"type": {hint_type}, '
            + (f'"possible_values": {possible_values},' if possible_values is not None else "")
            + f'"ctor": "{ctor}", '
            + f'"is_key": {self.to_bool(field.is_key)}, '
            + f'"is_array": {self.to_bool(is_sequence)}, '
            + f'"is_abstract": {self.to_bool(is_abstract)},'
            + f'"external": {self.to_bool(external)}}}'
        )

    @staticmethod
    def to_bool(_input: bool) -> str:
        return "True" if _input else "False"

    def get_mixins(self, struct_name: str) -> str:
        if f"{struct_name}Mixin" in self.mixins:
            return f"{struct_name}Mixin"

        if "AllMixin" in self.mixins:
            return "AllMixin"

        return ""

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

    def get_default_value(self, field: StructField):
        if isinstance(field.resolved_type, SequenceDefinition):
            return "[]"

        elif isinstance(field.resolved_type, ReferenceDefinition):
            return '""'

        elif isinstance(field, SimpleStructField):
            if field.type == "string" or field.type in self.string_types:
                assert not field.default or isinstance(field.default, str)
                return f'"{field.default if field.default else ""}"'

            elif field.type == "bool":
                assert not field.default or isinstance(field.default, bool)
                return "True" if field.default else "False"

            elif int_re.match(field.type):
                assert not field.default or isinstance(field.default, int)
                return f"{field.default if field.default else 0}"

            elif isinstance(field.resolved_type, EnumDefinition):
                assert not field.default or isinstance(field.default, str)
                return f'"{field.default if field.default else "Invalid"}"'

            assert not field.default, (
                "Currently `default:` is only allowed on simple types: "
                "integers, booleans, strings and enums."
            )

            if field.upcastable:
                return "None"
            else:
                return f"{field.type}()"

        else:
            raise ValueError()
