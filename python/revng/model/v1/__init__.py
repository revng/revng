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


class Binary(_generated.Binary):
    @classmethod
    def get_reference_str(cls, t):
        if hasattr(t, "Kind"):
            typename = t.Kind.value
        else:
            typename = type(t).__name__
        return f"/TypeDefinitions/{t.ID}-{typename}"


# Since we subclassed them we need to re-register their constructors and representers
YamlLoader.add_constructor("!Binary", Binary.yaml_constructor)
YamlDumper.add_representer(Binary, Binary.yaml_representer)
