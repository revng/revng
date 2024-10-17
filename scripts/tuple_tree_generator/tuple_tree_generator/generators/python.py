#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import ast
import os
from pathlib import Path
from typing import List

import black
from jinja2 import Environment

from ..schema import Definition, EnumDefinition, ReferenceDefinition, ScalarDefinition, Schema
from ..schema import SequenceDefinition, StructDefinition, StructField, UpcastableDefinition
from ..schema.struct import ReferenceStructField, SequenceStructField, SimpleStructField
from .jinja_utils import int_re, is_reference_struct_field, is_sequence_struct_field
from .jinja_utils import is_simple_struct_field, loader

BUILTINS_CLASSES = {"string": "str", "boolean": "bool", "int": "int"}


class PythonGenerator:
    def __init__(
        self,
        schema: Schema,
        root_type,
        output,
        string_types=None,
        external_types=None,
        mixins_paths=[],
    ):
        self.schema = schema
        self.root_type = root_type
        self.output = output
        self.string_types = string_types or []
        self.external_types = external_types or []
        self.mixins_paths = mixins_paths
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
        self.jinja_environment.filters["type_hint"] = self.type_hint
        self.jinja_environment.filters["docstring"] = self.render_docstring
        self.jinja_environment.filters["default_value"] = self.default_value
        self.jinja_environment.filters["gen_key"] = self.gen_key
        self.jinja_environment.filters["key_parser"] = self.key_parser
        self.jinja_environment.filters["get_mixins"] = self.get_mixins
        self.jinja_environment.globals["get_mixins_imports"] = self.get_mixins_imports
        self.jinja_environment.tests["simple_field"] = is_simple_struct_field
        self.jinja_environment.tests["sequence_field"] = is_sequence_struct_field
        self.jinja_environment.tests["reference_field"] = is_reference_struct_field
        self.template = self.jinja_environment.get_template("tuple_tree_gen.py.tpl")
        self.mixins: dict[str, List[str]] = self._parse_mixins()

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

    def scalar_converter(self, type_name: str) -> str:
        if type_name == "string":
            return "string"
        elif type_name == "bool":
            return "boolean"
        elif int_re.fullmatch(type_name):
            return "int"
        elif type_name in self.external_types + self.string_types:
            return type_name
        else:
            raise Exception(f"Unexpected scalar: {type_name}")

    def _real_type(self, field: StructField) -> str:
        resolved_type = field.resolved_type

        if isinstance(resolved_type, SequenceDefinition):
            resolved_type = resolved_type.element_type

        if isinstance(resolved_type, UpcastableDefinition):
            resolved_type = resolved_type.base

        real_type = resolved_type.name

        if isinstance(resolved_type, ScalarDefinition):
            real_type = self.scalar_converter(real_type)

        return real_type

    def type_hint(self, field: StructField) -> str:
        real_type = self._real_type(field)
        possible_values = None
        if real_type in BUILTINS_CLASSES:
            hint_type = BUILTINS_CLASSES[real_type]
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
        else:
            hint_type = real_type
            ctor = "class"

        assert hint_type

        is_sequence = isinstance(field.resolved_type, SequenceDefinition)
        if isinstance(field.resolved_type, SequenceDefinition):
            is_abstract = isinstance(field.resolved_type.element_type, UpcastableDefinition)
        else:
            is_abstract = isinstance(field.resolved_type, UpcastableDefinition)

        return (
            f'{{"type": {hint_type}, '
            + (f'"possibleValues": {possible_values},' if possible_values is not None else "")
            + f'"ctor": "{ctor}", '
            + f'"optional": {self.to_bool(field.optional)}, '
            + f'"isArray": {self.to_bool(is_sequence)}, '
            + f'"isAbstract": {self.to_bool(is_abstract)}}}'
        )

    @staticmethod
    def to_bool(_input: bool) -> str:
        return "True" if _input else "False"

    def _parse_mixins(self) -> dict[str, List[str]]:
        mixins: dict[str, List[str]] = {}
        for mixins_path in self.mixins_paths:
            with open(Path(mixins_path), "r") as mixin_file:
                mixins_parsed = ast.parse(mixin_file.read())
                mixin_classes = [
                    node.name for node in mixins_parsed.body if isinstance(node, ast.ClassDef)
                ]
                mixins[mixins_path] = mixin_classes
        return mixins

    def get_mixins(self, struct_name: str) -> str:
        for mixin_classes in self.mixins.values():
            for mc in mixin_classes:
                if struct_name == mc.removesuffix("Mixin"):
                    return mc

            if "AllMixin" in mixin_classes:
                return "AllMixin"
        raise RuntimeError("Couldn't find matching Mixin class")

    def get_mixins_imports(self) -> List[str]:
        """
        This method supports mixin imports from the `model` dir level or from subdirs
        """
        imports: List[str] = []
        for mixin_path, mixin_classes in self.mixins.items():
            path_distance = os.path.relpath(
                os.path.dirname(mixin_path), os.path.dirname(self.output)
            )
            file_name = os.path.basename(mixin_path).removesuffix(".py")

            if path_distance == ".":
                path = file_name
            elif path_distance == "..":
                path = f"..{file_name}"
            else:
                # Transform from unix path to python import
                path_distance = path_distance.replace("/", "", 1)
                path_distance = path_distance.replace("/", ".")
                path = f"{path_distance}.{file_name}"

            imports.append(f"from {path} import {', '.join(mixin_classes)}")  # noqa: E501
        return imports

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
