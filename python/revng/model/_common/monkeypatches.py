import random
from enum import Enum
from typing import Optional


def autoassign_constructor_argument(base_class: type, kwarg_name, kwarg_value):
    """Monkeypatches __init__ so that the given keyword argument is set to the given value if was not set by the caller"""
    assert isinstance(base_class, type)

    original_init = base_class.__init__

    def init_with_value(self, *args, **kwargs):
        if kwarg_name not in kwargs:
            kwargs[kwarg_name] = kwarg_value

        original_init(self, *args, **kwargs)

    base_class.__init__ = init_with_value


def autoassign_random_id(base_class: type):
    """Monkeypatches the __init__ method so that the ID is automatically assigned"""
    assert isinstance(base_class, type)

    original_init = base_class.__init__

    def init_with_random_id(self, *args, **kwargs):
        if "ID" not in kwargs:
            kwargs["ID"] = random.randint(2**10 + 1, 2**64 - 1)
        original_init(self, *args, **kwargs)

    base_class.__init__ = init_with_random_id


def autoassign_primitive_id(base_class: type):
    """Monkeypatches the __init__ method so that the ID is automatically assigned.
    Meant for use with primitive types.
    """
    assert isinstance(base_class, type)

    original_init = base_class.__init__

    def init_with_computed_id(self, *args, PrimitiveKind: "PrimitiveTypeKind", Size: int, **kwargs):
        if "ID" not in kwargs:
            primitive_kind_value = enum_value_to_index(PrimitiveKind)
            kwargs["ID"] = primitive_kind_value << 8 | Size
        original_init(self, *args, PrimitiveKind=PrimitiveKind, Size=Size, **kwargs)

    base_class.__init__ = init_with_computed_id


def enum_value_to_index(enum_value: Enum):
    """Converts an enum value to its index"""
    return list(enum_value.__class__.__members__).index(enum_value.value)


def make_hashable_using_attribute(base_type, attribute_name: Optional[str]):
    """Implements __hash__ by returning the given attribute"""

    def __hash__(self):
        if attribute_name is None:
            return id(self)
        return self.__getattribute__(attribute_name)

    base_type.__hash__ = __hash__
