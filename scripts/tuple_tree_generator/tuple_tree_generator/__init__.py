# This file is distributed under the MIT License. See LICENSE.md for details.

from .generators import CppHeadersGenerator, DocsGenerator, JSONSchemaGenerator, PythonGenerator
from .generators import TypeScriptGenerator
from .schema import Schema


def generate_cpp_headers(schema: Schema, root_type, user_include_path, emit_tracking):
    generator = CppHeadersGenerator(
        schema,
        root_type=root_type,
        emit_tracking=emit_tracking,
        user_include_path=user_include_path,
    )
    return generator.emit()


def generate_jsonschema(
    schema: Schema, root_type: str, string_types=None, separate_string_types=None
) -> str:
    generator = JSONSchemaGenerator(
        schema, root_type, string_types=string_types, separate_string_types=separate_string_types
    )
    return generator.emit_jsonschema()


def generate_python(schema: Schema, root_type: str, string_types=None, external_types=None) -> str:
    generator = PythonGenerator(
        schema,
        root_type,
        string_types=string_types,
        external_types=external_types,
    )
    return generator.emit_python()


def generate_typescript(*args, **kwargs) -> str:
    generator = TypeScriptGenerator(*args, **kwargs)
    return generator.emit_typescript()


def generate_docs(schema: Schema, root_type: str) -> str:
    generator = DocsGenerator(schema, root_type)
    return generator.emit_docs()
