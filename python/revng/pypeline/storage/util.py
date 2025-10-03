#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import graphlib

from revng.pypeline.object import Kind, ObjectID
from revng.pypeline.utils.registry import get_singleton

__all__ = ("_REVNG_VERSION_PLACEHOLDER", "_OBJECTID_MAXSIZE", "check_kind_structure")

_REVNG_VERSION_PLACEHOLDER = "1.0.0"

# Number of bytes that the biggest ObjectID will contain when serialized to bytes
# Currently fixed to 17 because 1 byte for kind + 16 bytes for MetaAddress
_OBJECTID_MAXSIZE = 17


# Check that the kind structure is smaller than the `_OBJECTID_MAXSIZE`. This
# is a hack to avoid having storage providers be more flexible about the
# structure of Kinds. Currently the storage providers can assume that the revng
# kind structure is the one they are operating on and make stronger assumptions
# and optimizations. If needed this check can be lifted but it would require
# extra work and extra flexibility.
def check_kind_structure():
    kind_type = get_singleton(Kind)
    kind_root = kind_type.root()
    kinds = kind_type.kinds()
    # Check that we can store all Kinds in a byte
    assert len(kinds) <= 2**8

    # Sort kinds
    sorter = graphlib.TopologicalSorter()
    for kind in kinds:
        if (parent := kind.parent()) is not None:
            sorter.add(kind, parent)
        else:
            sorter.add(kind)

    # Compute the serialized kind size
    kind_sizes: dict[Kind, int] = {}
    for kind in sorter.static_order():
        if kind == kind_root:
            kind_sizes[kind] = 0
        else:
            kind_sizes[kind] = kind_sizes[kind.parent()] + 1 + kind.byte_size()

    # Check that the maximum kind size does not exceed the expected maximum size
    assert max(kind_sizes.values()) <= _OBJECTID_MAXSIZE


# Generally SQL-like implementations require, in the general case, to
# dynamically generate the query to invalidate objects from paths. To simplify
# things we rely on the fact that currently objects have a maximum depth of
# 2. This function will be checked by SQL-like StorageProviders when adding
# object dependencies via `add_dependencies`.
def check_object_id_supported_by_sql(object_id: ObjectID):
    if object_id.kind().rank() >= 2:
        raise NotImplementedError("Objects with rank >= 2 are not supported")
