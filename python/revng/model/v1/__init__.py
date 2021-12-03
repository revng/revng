#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from ._generated import *
from .metaaddress import MetaAddress, MetaAddressType
from .reference import Reference
from .._common.monkeypatches import (
    autoassign_primitive_id,
    autoassign_random_id,
    autoassign_constructor_argument,
    make_hashable_using_attribute,
)

# Automatically compute primitive type ID if we weren't provided one
autoassign_primitive_id(Primitive)

# Automatically assign a random ID if we weren't provided one
types_with_random_id = [
    CABIFunctionType,
    RawFunctionType,
    UnionType,
    Struct,
    EnumType,
    Typedef,
]

for t in types_with_random_id:
    autoassign_random_id(t)

# Autoassign the Kind constructor argument
types_to_kind = [
    (Primitive, TypeKind.Primitive),
    (EnumType, TypeKind.Enum),
    (Typedef, TypeKind.Typedef),
    (Struct, TypeKind.Struct),
    (UnionType, TypeKind.Union),
    (CABIFunctionType, TypeKind.CABIFunctionType),
    (RawFunctionType, TypeKind.RawFunctionType),
]
for t, kind_val in types_to_kind:
    autoassign_constructor_argument(t, "Kind", kind_val)

# Implement __hash__ based on the types ID
hashable_types = [
    (CABIFunctionType, "ID"),
    (RawFunctionType, "ID"),
    (UnionType, "ID"),
    (Struct, "ID"),
    (EnumType, "ID"),
    (Typedef, "ID"),
    (Primitive, "ID"),
    (Function, None),
]

for t, attr_name in hashable_types:
    make_hashable_using_attribute(t, attr_name)
