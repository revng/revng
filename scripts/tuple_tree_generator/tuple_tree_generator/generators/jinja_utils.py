#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from pathlib import Path

import jinja2

from ..schema.struct import ReferenceStructField, SequenceStructField, SimpleStructField

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

loader = jinja2.FileSystemLoader(str(TEMPLATES_DIR))


environment = jinja2.Environment(
    block_start_string="/**",
    block_end_string="**/",
    variable_start_string="/*=",
    variable_end_string="=*/",
    comment_start_string="/*#",
    comment_end_string="#*/",
    loader=loader,
)

python_environment = jinja2.Environment(
    block_start_string="##",
    block_end_string="##",
    variable_start_string="'",
    variable_end_string="'",
    comment_start_string="###",
    comment_end_string="###",
    loader=loader,
)


def render_docstring(docstr: str):
    if not docstr:
        return ""
    rendered_docstring = "\n".join(f"/// {line}" for line in docstr.splitlines())
    if not rendered_docstring.endswith("\n"):
        rendered_docstring = rendered_docstring + "\n"
    return rendered_docstring


environment.filters["docstring"] = render_docstring

# More convenient than escaping the double braces
environment.globals["nodiscard"] = "[[ nodiscard ]]"


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


def is_simple_struct_field(obj):
    return isinstance(obj, SimpleStructField)


def is_sequence_struct_field(obj):
    return isinstance(obj, SequenceStructField)


def is_reference_struct_field(obj):
    return isinstance(obj, ReferenceStructField)


python_environment.filters["docstring"] = render_python_docstring
python_environment.tests.update(
    {
        "simple_field": is_simple_struct_field,
        "sequence_field": is_sequence_struct_field,
        "reference_field": is_reference_struct_field,
    }
)
