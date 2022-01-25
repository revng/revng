from typing import IO, Text

import yaml

from ._generated import (
    Architecture,
    Binary,
    CABIFunctionType,
    ABI,
    Argument,
    RawFunctionType,
    NamedTypedRegister,
    TypedRegister,
    Register,
    Struct,
    StructField,
    UnionModel as Union,
    UnionField,
    EnumModel as Enum,
    EnumEntry,
    Typedef,
    TypeKind,
    PrimitiveTypeKind,
    QualifiedType,
    Qualifier,
    QualifierKind,
    Function,
    FunctionType,
    BasicBlock,
    CallEdge,
    FunctionEdge,
    FunctionEdgeType,
    Segment,
    DynamicFunction,
    Primitive,
)
from .base import YamlLoader, YamlDumper
from .metaaddress import MetaAddress, MetaAddressType
from .typeref import Typeref
from .._common.monkeypatches import autoassign_primitive_id, autoassign_random_id, autoassign_constructor_argument, make_hashable_using_attribute

# Automatically compute primitive type ID if we weren't provided one
autoassign_primitive_id(Primitive)

# Automatically assign a random ID if we weren't provided one
types_with_random_id = [
    CABIFunctionType,
    RawFunctionType,
    Union,
    Struct,
    Enum,
    Typedef,
]

for t in types_with_random_id:
    autoassign_random_id(t)

# Autoassign the Kind constructor argument
types_to_kind = [
    (Primitive, TypeKind.Primitive),
    (Enum, TypeKind.Enum),
    (Typedef, TypeKind.Typedef),
    (Struct, TypeKind.Struct),
    (Union, TypeKind.Union),
    (CABIFunctionType, TypeKind.CABIFunctionType),
    (RawFunctionType, TypeKind.RawFunctionType),
]
for t, kind_val in types_to_kind:
    autoassign_constructor_argument(t, "Kind", kind_val)

# Implement __hash__ based on the types ID
hashable_types = [
    (CABIFunctionType, "ID"),
    (RawFunctionType, "ID"),
    (Union, "ID"),
    (Struct, "ID"),
    (Enum, "ID"),
    (Typedef, "ID"),
    (Primitive, "ID"),
    (Function, None),
]

for t, attr_name in hashable_types:
    make_hashable_using_attribute(t, attr_name)


def load_model(stream: bytes | IO[bytes] | Text | IO[Text]):
    serialized_model = yaml.load(stream, Loader=YamlLoader)
    return Binary.parse_obj(serialized_model)
