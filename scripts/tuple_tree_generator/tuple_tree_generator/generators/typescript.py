#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path
from typing import List, Optional

from jinja2 import Environment
from markupsafe import Markup

from ..schema import EnumDefinition, ReferenceDefinition, ScalarDefinition, Schema
from ..schema import SequenceDefinition, StructDefinition, StructField, UpcastableDefinition
from ..schema.struct import SimpleStructField
from .jinja_utils import int_re, loader

BUILTINS = ["string", "boolean", "bigint"]
BUILTINS_CLASSES = {"string": "String", "boolean": "Boolean", "bigint": "BigIntBuilder"}


class TypeScriptGenerator:
    def __init__(
        self,
        schema: Schema,
        root_type: str,
        global_name: str,
        external_files: Optional[List[str]] = None,
        string_types: Optional[List[str]] = None,
        external_types: Optional[List[str]] = None,
    ):
        self.schema = schema
        self.metadata = {"root": root_type, "namespace": schema.base_namespace}
        self.string_types = string_types if string_types is not None else []
        self.external_types = external_types if external_types is not None else []
        self.external_files = external_files if external_files is not None else []
        self.global_name = global_name
        self.jinja_environment = Environment(loader=loader)
        self.jinja_environment.globals["load_file"] = self.load_file
        self.jinja_environment.globals["is_optional"] = self.is_optional
        self.jinja_environment.globals["completely_optional"] = self.completely_optional
        self.jinja_environment.globals["default_value"] = self.get_default_value
        self.jinja_environment.filters["read_file"] = self.read_file
        self.jinja_environment.filters["ts_doc"] = self.ts_doc
        self.jinja_environment.filters["ts_type"] = self.ts_type
        self.jinja_environment.filters["ts_itype"] = self.ts_itype
        self.jinja_environment.filters["get_guid"] = self.get_guid_field
        self.jinja_environment.filters["get_attr"] = self.get_attr
        self.jinja_environment.filters["gen_assignment"] = self.gen_assignment
        self.jinja_environment.filters["gen_key"] = self.gen_key
        self.jinja_environment.filters["key_parser"] = self.key_parser
        self.jinja_environment.filters["type_hint"] = self.type_hint

    def emit_typescript(self) -> str:
        """Returns typescript definitions as a string"""

        return self.jinja_environment.get_template("tuple_tree_gen.ts.tpl").render(
            enums=self.schema.enum_definitions(),
            structs=self.schema.struct_definitions(),
            metadata=self.metadata,
            external_files=self.external_files,
            global_name=self.global_name,
            string_types=self.string_types,
            external_types=self.external_types,
        )

    def load_file(self, name: str):
        source, _, _ = loader.get_source(self.jinja_environment, name)
        source_no_license = "\n".join(source.split("\n")[3:])
        return Markup(source_no_license)

    @staticmethod
    def get_attr(obj, attr_name):
        return getattr(obj, attr_name, None)

    def ts_type(self, field: StructField) -> str:
        if isinstance(field.resolved_type, ReferenceDefinition):
            pointee_type = field.resolved_type.pointee.name
            root_type = field.resolved_type.root.name
            return f"Reference<{pointee_type},{root_type}>"
        elif isinstance(field.resolved_type, SequenceDefinition):
            real_type = self._real_type(field)
            return f"{real_type}[]"
        else:
            return self._real_type(field)

    def ts_itype(self, field: StructField) -> str:
        if self._is_simple_type(field):
            return self.ts_type(field)
        if isinstance(field.resolved_type, ReferenceDefinition):
            return "IReference"
        return f"I{self.ts_type(field)}"

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

    def _is_simple_type(self, field: StructField) -> bool:
        """If True the field is either a builtin or enum or a sequence of them"""
        real_type = self._real_type(field)
        return (real_type in BUILTINS) or (
            real_type in [e.name for e in self.schema.enum_definitions()]
        )

    def _is_class_type(self, field: StructField, only_concrete=False) -> bool:
        real_type = self._real_type(field)
        cls_builtins = self.external_types + (self.string_types if not only_concrete else [])
        return (real_type in cls_builtins) or (
            real_type
            in [
                s.name
                for s in self.schema.struct_definitions()
                if (not only_concrete) or (not s.abstract)
            ]
        )

    @classmethod
    def get_guid_field(cls, class_: StructDefinition) -> Optional[StructField]:
        return next((field for field in class_.fields if field.is_guid), None)

    def gen_assignment(self, field: StructField) -> str:
        if self._is_simple_type(field):
            result = f"this.{field.name} = rawObject.{field.name}"
            opt = f"|| {self.get_default_value(field)}" if field.optional else ""
            return f"{result}{opt};"
        if isinstance(field.resolved_type, ReferenceDefinition):
            pointee_type = field.resolved_type.pointee.name
            root_type = field.resolved_type.root.name
            return (
                f"this.{field.name} = "
                + f"Reference.parse<{pointee_type},{root_type}>(rawObject.{field.name});"
            )
        if self._is_class_type(field):
            is_sequence = isinstance(field.resolved_type, SequenceDefinition)
            ftype = self._real_type(field)
            name = field.name
            if self._is_class_type(field, True):
                if is_sequence:
                    return f"this.{name} = (rawObject.{name} || []).map(e => new {ftype}(e));"
                else:
                    return f"this.{name} = new {ftype}(rawObject.{name});"
            else:
                if is_sequence:
                    return f"this.{name} = (rawObject.{name} || []).map(e => {ftype}.parse(e));"
                else:
                    return f"this.{name} = {ftype}.parse(rawObject.{name});"
        raise ValueError(field.name)

    def get_default_value(self, field: StructField):
        if isinstance(field.resolved_type, SequenceDefinition):
            return "[]"
        elif isinstance(field.resolved_type, ReferenceDefinition):
            return 'new Reference("")'
        elif isinstance(field, SimpleStructField):
            if field.type == "string":
                return '""'
            elif field.type in self.string_types:
                return f'new {field.type}("")'
            elif field.type == "bool":
                return "false"
            elif int_re.match(field.type):
                return "0n"
            else:
                if isinstance(field.resolved_type, EnumDefinition):
                    return '"Invalid"'
                else:
                    return f"new {field.type}()"
        else:
            raise ValueError()

    @staticmethod
    def gen_key(struct: StructDefinition) -> str:
        fields = [f"${{this.{f.name}}}" for f in struct.key_fields]
        return f"`{'-'.join(fields)}`"

    @staticmethod
    def key_parser(struct: StructDefinition) -> str:
        fields = [f"{f.name}: parts[{i}]" for i, f in enumerate(struct.key_fields)]
        return ",".join(fields)

    @classmethod
    def ts_doc(cls, doc: str) -> str:
        if doc is not None:
            return cls._render_doc(doc)
        else:
            return ""

    @staticmethod
    def _render_doc(doc: str) -> str:
        lines = [x for x in doc.split("\n") if x.strip() != ""]
        if len(lines) > 1:
            doc_lines = "\n".join([f"* {x}" for x in lines])
            return f"""
                /**
                {doc_lines}
                 */"""
        elif len(lines) == 1:
            return f"// {lines[0]}"
        else:
            return ""

    def scalar_converter(self, type_name: str) -> str:
        if type_name == "string":
            return "string"
        elif type_name == "bool":
            return "boolean"
        elif int_re.fullmatch(type_name):
            return "bigint"
        elif type_name in self.external_types + self.string_types:
            return type_name
        else:
            raise Exception(f"Unexpected scalar: {type_name}")

    @staticmethod
    def read_file(path: str) -> str:
        file_path = Path(path)
        if file_path.is_file():
            source = file_path.read_text(encoding="utf-8")
            source_no_license = "\n".join(source.split("\n")[3:])
            return Markup(source_no_license)
        else:
            raise ValueError()

    def type_hint(self, field: StructField) -> str:
        real_type = self._real_type(field)
        possible_values = None
        if real_type in BUILTINS:
            hint_type = BUILTINS_CLASSES.get(real_type)
            ctor = "native"
        elif real_type in [e.name for e in self.schema.enum_definitions()]:
            hint_type = "String"
            ctor = "enum"
            possible_values = f"{real_type}Values"
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

        return (
            f"{{type: {hint_type}, "
            + (f"possibleValues: {possible_values}," if possible_values is not None else "")
            + f"ctor: '{ctor}', "
            + f"optional: {'true' if field.optional else 'false'}, "
            + f"isArray: {'true' if is_sequence else 'false'}}}"
        )

    @staticmethod
    def is_optional(field: StructField):
        return (
            field.optional or field.is_guid or isinstance(field.resolved_type, ReferenceDefinition)
        )

    @classmethod
    def completely_optional(cls, class_: StructDefinition):
        return all(cls.is_optional(f) for f in class_.fields)
