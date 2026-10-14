#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase

# The only floating point primitives the model now accepts are C's `float` and
# `double`, i.e. 4- and 8-byte ones only.
# More used to be allowed, so models produced before this restriction may
# contain them.
#
# `Generic` accepts every such size, so widening the primitive's *kind* is enough
# to make the model valid again while preserving the layout: the size, and
# therefore every offset and alignment computed from it, is untouched.

_VALID_FLOAT_SIZES = (4, 8)


def _demote_float_primitives(node):
    """Rewrite invalid `Float` primitives into `Generic` ones, in place.

    Note that this has to walk the entire model - as a primitive can appear as
    a leaf anywhere in the type system.
    """
    if isinstance(node, dict):
        if (
            node.get("Kind") == "PrimitiveType"
            and node.get("PrimitiveKind") == "Float"
            and node.get("Size") not in _VALID_FLOAT_SIZES
        ):
            node["PrimitiveKind"] = "Generic"
        for value in node.values():
            _demote_float_primitives(value)

    elif isinstance(node, list):
        for value in node:
            _demote_float_primitives(value)


class Migration(MigrationBase):
    def migrate(self, binary):
        _demote_float_primitives(binary)
