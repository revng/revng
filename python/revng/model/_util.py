#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from enum import Enum


def enum_value_to_index(enum_value: Enum):
    """Converts an enum value to its index"""
    return list(enum_value.__class__.__members__).index(enum_value.value)


# TODO: once tuple_tree_generator becomes kind-aware, this will no longer be necessary
def force_constructor_kwarg(base_class: type, kwarg_name, kwarg_value):
    """Monkeypatches the __init__ method so that the given kwarg is forced to the given value.
    If the argument was not provided it is set, if it was provided and it was different from the forced value a
    ValueError is raised.
    """
    assert isinstance(base_class, type)

    original_init = base_class.__init__

    def init_forcing_value(self, *args, **kwargs):
        if kwarg_name not in kwargs:
            kwargs[kwarg_name] = kwarg_value
        else:
            if kwargs[kwarg_name] != kwarg_value:
                raise ValueError(f"Invalid value provided for {kwarg_name}")

        original_init(self, *args, **kwargs)

    base_class.__init__ = init_forcing_value
