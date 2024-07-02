#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# flake8: noqa: F405
# type: ignore

from io import TextIOBase
from typing import Optional, Union

import yaml

from revng.model.metaaddress import MetaAddress, MetaAddressType, init_metaaddress_yaml_classes
from revng.tupletree import DiffSet as TTDiffSet
from revng.tupletree import Reference, StructBase, TypesMetadata, _get_element_by_path
from revng.tupletree import enum_value_to_index, init_reference_yaml_classes

from . import _generated
from ._generated import *
from .metaaddress import *

init_metaaddress_yaml_classes(YamlLoader, YamlDumper)
init_reference_yaml_classes(YamlLoader, YamlDumper)

init_metaaddress_yaml_classes(DiffYamlLoader, DiffYamlDumper)
init_reference_yaml_classes(DiffYamlLoader, DiffYamlDumper)


class Binary(_generated.Binary):
    @classmethod
    def get_reference_str(cls, t):
        if hasattr(t, "Kind"):
            typename = t.Kind.value
        else:
            typename = type(t).__name__
        return f"/TypeDefinitions/{t.ID}-{typename}"

    @staticmethod
    def deserialize(input_: Union[str, TextIOBase]) -> Binary:
        return yaml.load(input_, Loader=YamlLoader)

    def serialize(self, output: Optional[TextIOBase] = None):
        if output is None:
            return yaml.dump(self, Dumper=YamlDumper)
        else:
            return yaml.dump(self, output, Dumper=YamlDumper)


class DiffSet(TTDiffSet[Binary]):
    @staticmethod
    def _get_root() -> StructBase:
        return _generated.Binary

    @staticmethod
    def _get_types_metadata() -> TypesMetadata:
        return _generated.types_metadata

    @classmethod
    def deserialize(cls, input_: Union[str, TextIOBase]) -> "DiffSet":
        diff_dict = yaml.load(input_, Loader=DiffYamlLoader)
        diff = cls.from_dict(diff_dict)
        return _parse_diff(diff, cls._get_type_info)

    def serialize(self, output: Optional[TextIOBase] = None):
        if output is None:
            return yaml.dump(self, Dumper=DiffYamlDumper)
        else:
            yaml.dump(self, output, Dumper=DiffYamlDumper)


def get_element_by_path(path: str, obj: StructBase) -> StructBase:
    return _get_element_by_path(path, obj)


# Since we subclassed them we need to re-register their constructors and representers
YamlLoader.add_constructor("!Binary", Binary.yaml_constructor)
YamlDumper.add_representer(Binary, Binary.yaml_representer)

DiffYamlLoader.add_constructor("!DiffSet", DiffSet.yaml_constructor)
DiffYamlDumper.add_representer(DiffSet, DiffSet.yaml_representer)
