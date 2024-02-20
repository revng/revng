#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# flake8: noqa: F405
# type: ignore

from revng.tupletree import Reference, enum_value_to_index, init_reference_yaml_classes

from ..metaaddress import *
from . import _generated
from ._generated import *

init_metaaddress_yaml_classes(YamlLoader, YamlDumper)
init_reference_yaml_classes(YamlLoader, YamlDumper)


# Create a derived PrimitiveDefinition class that automatically computes the correct ID
class PrimitiveDefinition(_generated.PrimitiveDefinition):
    def __init__(self, *args, **kwargs):
        if "ID" not in kwargs:
            primitive_kind = kwargs.get("PrimitiveKind")
            if primitive_kind is None:
                raise ValueError(
                    "Must provide PrimitiveKind as kwarg to PrimitiveDefinition constructor"
                )

            size = kwargs.get("Size")
            if size is None:
                raise ValueError("Must provide Size as kwarg to PrimitiveDefinition constructor")

            primitive_kind_value = enum_value_to_index(primitive_kind)
            kwargs["ID"] = primitive_kind_value << 8 | size

        super(self.__class__, self).__init__(*args, **kwargs)


class Binary(_generated.Binary):
    @classmethod
    def get_reference_str(cls, t):
        if hasattr(t, "Kind"):
            typename = t.Kind.value
        else:
            typename = type(t).__name__
        return f"/TypeDefinitions/{t.ID}-{typename}"


# Since we subclassed them we need to re-register their constructors and representers
YamlDumper.add_representer(PrimitiveDefinition, PrimitiveDefinition.yaml_representer)
YamlLoader.add_constructor("!Binary", Binary.yaml_constructor)
YamlDumper.add_representer(Binary, Binary.yaml_representer)
