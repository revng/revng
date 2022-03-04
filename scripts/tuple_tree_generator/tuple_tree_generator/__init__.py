# This file is distributed under the MIT License. See LICENSE.md for details.

from .generators import CppHeadersGenerator
from .generators import JSONSchemaGenerator
from .generators import PythonGenerator
from .schema import Schema


def generate_cpp_headers(schema: Schema, user_include_path):
    generator = CppHeadersGenerator(schema, user_include_path=user_include_path)
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
