#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase

# The only floating point primitives the model now accepts are C's `float` and
# `double`, i.e. 4- and 8-byte ones only.
# More used to be allowed, so models produced before this restriction may
# contain them.
#
# For primitives that still have a `Generic` equivalent (namely the 2-byte and
# 16-byte ones), just set the kind to `Generic`.
#
# For the primitives that are completely removed (namely the 10- and 12-byte
# ones), do nothing - pretend they don't exist.
#
# TODO: handle this in a clearer fashion.

_VALID_FLOAT_SIZES = (4, 8)
_VALID_GENERIC_SIZES = (1, 2, 4, 8, 16, 32, 64)


def _demote_float_primitives(node):
    """Rewrite invalid `Float` primitives into `Generic` ones, in place.

    Primitives whose size no kind accepts are left untouched.

    Note that this has to walk the entire model - as a primitive can appear as
    a leaf anywhere in the type system.
    """
    if isinstance(node, dict):
        if (
            node.get("Kind") == "PrimitiveType"
            and node.get("PrimitiveKind") == "Float"
            and node.get("Size") not in _VALID_FLOAT_SIZES
            and node.get("Size") in _VALID_GENERIC_SIZES
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
