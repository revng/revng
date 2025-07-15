#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# flake8: noqa: F405
# type: ignore

from dataclasses import dataclass
from io import TextIOBase
from typing import Optional, Union

import yaml

from revng.model.metaaddress import MetaAddress, MetaAddressType, init_metaaddress_yaml_classes
from revng.tupletree import DiffSet as TTDiffSet
from revng.tupletree import Reference, StructBase, TypesMetadata, _FieldVisitor
from revng.tupletree import _get_element_by_path, init_reference_yaml_classes

from . import _generated
from ._generated import *
from .metaaddress import *

init_metaaddress_yaml_classes(YamlLoader, YamlDumper)
init_reference_yaml_classes(YamlLoader, YamlDumper)

init_metaaddress_yaml_classes(DiffYamlLoader, DiffYamlDumper)
init_reference_yaml_classes(DiffYamlLoader, DiffYamlDumper)


class Binary(_generated.Binary):
    @staticmethod
    def deserialize(input_: Union[str, TextIOBase]) -> Binary:
        obj = yaml.load(input_, Loader=YamlLoader)
        _fix_references(obj)
        return obj


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


@dataclass
class TypeId:
    id_: int
    kind: _generated.TypeDefinitionKind

    def to_string(self):
        return f"{self.id_}-{self.kind}"


def get_element_by_path(path: str, obj: StructBase) -> StructBase:
    return _get_element_by_path(path, obj)


def iterate_fields(obj: Binary):
    return _FieldVisitor(_generated.types_metadata, _generated.Binary).visit(obj)


def _fix_references(obj: Binary):
    accessor = lambda x: get_element_by_path(x, obj)
    for _, field_obj in iterate_fields(obj):
        if not isinstance(field_obj, Reference) or not field_obj.is_valid():
            continue
        field_obj._accessor = accessor


# Since we subclassed them we need to re-register their constructors and representers
YamlLoader.add_constructor("!Binary", Binary.yaml_constructor)
YamlDumper.add_representer(Binary, Binary.yaml_representer)

DiffYamlLoader.add_constructor("!DiffSet", DiffSet.yaml_constructor)
DiffYamlDumper.add_representer(DiffSet, DiffSet.yaml_representer)
