#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# flake8: noqa: F405
# type: ignore

from revng.tupletree import DiffSet, Reference, StructBase, _apply_diff, _get_element_by_path
from revng.tupletree import _get_parent_element_by_path, _get_type_info, _make_diff, _parse_diff
from revng.tupletree import _validate_diff, enum_value_to_index, init_reference_yaml_classes

from ..metaaddress import *
from . import _generated
from ._generated import *

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


def make_diff(obj_old: StructBase, obj_new: StructBase) -> DiffSet:
    return _make_diff(obj_old, obj_new, TypeHints, _generated.Binary)


def get_type_info(path: str, root=_generated.Binary) -> StructBase:
    return _get_type_info(path, root, TypeHints)


def get_element_by_path(path: str, obj: StructBase) -> StructBase:
    return _get_element_by_path(path, obj)


def get_parent_element_by_path(path: str, obj: StructBase) -> StructBase:
    return _get_parent_element_by_path(path, obj)


def validate_diff(obj: StructBase, diff: DiffSet) -> bool:
    return _validate_diff(obj, diff, get_type_info)


def apply_diff(obj: StructBase, diff: DiffSet) -> tuple[bool, StructBase | None]:
    return _apply_diff(obj, diff, validate_diff)


# Since we subclassed them we need to re-register their constructors and representers
YamlLoader.add_constructor("!Binary", Binary.yaml_constructor)
YamlDumper.add_representer(Binary, Binary.yaml_representer)

DiffYamlLoader.add_constructor("!DiffSet", DiffSet.yaml_constructor)
DiffYamlDumper.add_representer(DiffSet, DiffSet.yaml_representer)


def load_model(path: str) -> Binary:
    with open(path) as f:
        return yaml.load(f, Loader=YamlLoader)


def load_model_from_string(model: str) -> Binary:
    return yaml.load(model, Loader=YamlLoader)


def dump_model(path: str, obj: StructBase) -> yaml:
    with open(path, "w") as f:
        yaml.dump(obj, f, Dumper=YamlDumper)


def load_diff(path: str) -> DiffSet:
    with open(path) as f:
        diff = yaml.load(f, Loader=DiffYamlLoader)
        return _parse_diff(diff, get_type_info)


def dump_diff(path, diff) -> yaml:
    with open(path, "w") as f:
        yaml.dump(diff, f, Dumper=DiffYamlDumper)


def dump_diff_to_string(diff) -> yaml:
    return yaml.dump(diff, Dumper=DiffYamlDumper)
