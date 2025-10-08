#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.pypeline.object import Kind
from revng.pypeline.utils.registry import get_singleton

__all__ = ("_REVNG_VERSION_PLACEHOLDER", "_KIND_MAXDEPTH", "check_kind_structure")

_REVNG_VERSION_PLACEHOLDER = "1.0.0"

# Number of bytes that the biggest ObjectID will contain when serialized to bytes
_KIND_MAXDEPTH = 17


def check_kind_structure():
    kind_type = get_singleton(Kind)
    kind_root = kind_type.root()
    kinds = kind_type.kinds()
    # Check that we can store all Kinds in a byte
    assert len(kinds) <= 2**8

    # Check that each byte does not exceed the mask size
    for kind in kinds:
        assert kind.byte_size() <= (_KIND_MAXDEPTH - 1)
        assert kind == kind_root or kind.parent() == kind_root
