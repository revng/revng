#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import black

from ..schema import Schema
from .jinja_utils import python_environment

template = python_environment.get_template("tuple_tree_gen.py.tpl")


class PythonGenerator:
    def __init__(self, schema: Schema, root_type, string_types=None, external_types=None):
        self.schema = schema
        self.root_type = root_type
        self.string_types = string_types or []
        self.external_types = external_types or []

    def emit_python(self) -> str:
        rendered_template = template.render(
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
