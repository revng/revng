#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


def _make_primitive_sizes_explicit(node):
    if isinstance(node, dict):
        if node.get("Kind") == "PrimitiveType":
            node.setdefault("Size", 0)
        for value in node.values():
            _make_primitive_sizes_explicit(value)
    elif isinstance(node, list):
        for value in node:
            _make_primitive_sizes_explicit(value)


class Migration(MigrationBase):
    """Make the size of primitive types explicit.

    Absent fields take the default value declared in the schema, which, for
    `PrimitiveType.Size`, is -1. In previous versions, an absent `Size`
    (e.g., in `void`) stood for 0.
    """

    def migrate(self, binary):
        _make_primitive_sizes_explicit(binary)
