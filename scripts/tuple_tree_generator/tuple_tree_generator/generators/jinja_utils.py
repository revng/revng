#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from pathlib import Path

import jinja2

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
