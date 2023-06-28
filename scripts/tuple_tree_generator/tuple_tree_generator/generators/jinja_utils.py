#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import re
from pathlib import Path

import jinja2

from ..schema.struct import ReferenceStructField, SequenceStructField, SimpleStructField

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

loader = jinja2.FileSystemLoader(str(TEMPLATES_DIR))
int_re = re.compile(r"(u)?int(8|16|32|64)_t")


def is_simple_struct_field(obj):
    return isinstance(obj, SimpleStructField)


def is_sequence_struct_field(obj):
    return isinstance(obj, SequenceStructField)


def is_reference_struct_field(obj):
    return isinstance(obj, ReferenceStructField)
