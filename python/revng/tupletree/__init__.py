#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
from collections.abc import MutableSequence
from copy import deepcopy
from dataclasses import dataclass, fields
from enum import Enum, EnumType
from functools import lru_cache
from typing import Any, Callable, Dict, Generic, List, Type, TypeVar, get_args, get_origin
from typing import get_type_hints

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CSafeLoader as Loader
except ImportError:
    sys.stderr.write("Warning: using the slow pure-python YAML loader and dumper!\n")
    from yaml import Dumper  # type: ignore
    from yaml import SafeLoader as Loader  # type: ignore

no_default = object()

dataclass_kwargs = {}
if sys.version_info >= (3, 10, 0):
    # Performance optimization available since python 3.10
    dataclass_kwargs["slots"] = True
    dataclass_kwargs["kw_only"] = True


def _create_instance(field_value, field_type):
    # First we check if the value is already of the required type
    if isinstance(field_value, field_type):
        return field_value
    if isinstance(field_value, str) and issubclass(field_type, Enum):
        return field_type(field_value)
    if isinstance(field_value, dict) and issubclass(field_type, StructBase):
        return field_type.from_dict(**field_value)
    if isinstance(field_value, str) and hasattr(field_type, "from_string"):
        return field_type.from_string(field_value)

    raise TypeError(f"Invalid type {type(field_value)}, was expecting {field_type}")


# Hot function, hence the lru_cache
# Called once per field for each object instantiation
@lru_cache(maxsize=1024, typed=True)
def get_type_hint_cached(class_, name):
    return get_type_hints(class_)[name]


@dataclass
class StructBase:
    @classmethod
    def from_dict(cls, **kwargs):
        """Constructs an instance of the object using the values supplied as kwargs"""
        constructor_kwargs = {}

        # Iterate over all the fields defined in the dataclass
        for field_name, field_value in kwargs.items():
            field_spec = cls.__dataclass_fields__.get(field_name)
            if field_spec is None:
                raise ValueError(f"Field {field_name} is not allowed for type {cls.__name__}")

            # Get the type annotation
            field_spec_type = get_type_hint_cached(cls, field_spec.name)
            # Get the "origin", i.e. for a field annotated as List[str] the origin is list
            origin = get_origin(field_spec_type)
            # Get the args, i.e. for a field annotated as Dict[str, int] the args are (str, int)
            args = get_args(field_spec_type)

            # If the field is a list of something we need to instantiate its elements one by one
            if origin is list:
                assert len(args) == 1
                underlying_type = args[0]

                if not isinstance(field_value, list):
                    raise ValueError(
                        f"Expected list for field {field_name} of {cls.__name__},"
                        + f"got {type(field_value)}"
                    )

                instances = []
                for v in field_value:
                    try:
                        v_inst = _create_instance(v, underlying_type)
                        instances.append(v_inst)
                    except ValueError as e:
                        raise ValueError(
                            f"Error deserializing list element of {field_name} of {cls.__name__}"
                        ) from e

                constructor_kwargs[field_name] = instances

            elif origin is Reference:
                constructor_kwargs[field_name] = Reference(field_value)

            # The field is not a list nor a reference, create an instance of the field value
            else:
                try:
                    constructor_kwargs[field_name] = _create_instance(field_value, field_spec_type)
                except ValueError as e:
                    raise TypeError(
                        f"Error while deserializing field {field_name} of {cls.__name__}"
                    ) from e

        instance = cls(**constructor_kwargs)
        return instance

    @classmethod
    def from_string(cls, s):
        raise NotImplementedError(f"from_string not implemented for {cls.__name__}")

    @classmethod
    def get_reference_str(cls, obj):
        # Types that can be considered "roots" of a tupletree must implement this method
        raise NotImplementedError(f"get_reference_str not implemented for {cls.__name__}")

    @classmethod
    def yaml_constructor(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return cls.from_dict(**mapping)

    @classmethod
    def yaml_representer(cls, dumper: yaml.dumper.Dumper, instance):
        mapping_to_dump = {}
        for field in fields(cls):
            field_val = instance.__getattribute__(field.name)
            if _field_is_default(field, field_val) or (
                isinstance(field_val, Reference) and not field_val.is_valid()
            ):
                continue
            mapping_to_dump[field.name] = field_val
        return dumper.represent_dict(mapping_to_dump)

    def __post_init__(self):
        # Before python 3.10 dataclasses had an annoying limitation regarding inheritance:
        # if the base class has a field with a default value, *all* the fields of the child classes
        # need to have a default value too, or dataclasses will raise a TypeError.
        # Since python 3.10 this limitation can be mostly bypassed by defining the base class
        # fields as kw_only, but we want to support older python versions.
        # Hence this workaround, inspired by https://stackoverflow.com/a/53085935
        for field in fields(self):
            field_value = self.__getattribute__(field.name)
            field_hints = get_type_hint_cached(self.__class__, field.name)
            if field_value is no_default:
                raise TypeError(f"__init__ missing 1 required argument: {field.name}")
            if get_origin(field_hints) is list:
                new_field_value = TypedList(get_args(field_hints)[0])
                new_field_value.extend(field_value)
                setattr(self, field.name, new_field_value)

    def __setattr__(self, key, value):
        # Prevent setting undefined attributes, `_` allows to set private attributes like `_project`
        if not (key in [f.name for f in fields(self)] or key.startswith("_")):
            raise AttributeError(f"Cannot set attribute {key} for class {type(self).__name__}")
        super().__setattr__(key, value)


class AbstractStructBase(StructBase):
    _children: Dict[str, Type] = {}

    @classmethod
    def from_dict(cls, **kwargs):
        if "Kind" not in kwargs:
            raise ValueError("Upcastable types must have a Kind field")

        child_cls = cls._children.get(kwargs["Kind"])

        if not child_cls:
            raise ValueError(f"No class found to deserialize {kwargs['Kind']}")

        if cls != child_cls:
            return child_cls.from_dict(**kwargs)
        return super().from_dict(**kwargs)


class DefaultEnumType(EnumType):
    default = object()

    def __call__(cls, value=default, *args, **kwargs):  # noqa: N805
        if value is DefaultEnumType.default:
            return cls.Invalid
        return super().__call__(value, *args, **kwargs)


class EnumBase(Enum, metaclass=DefaultEnumType):
    @classmethod
    def yaml_representer(cls, dumper: yaml.dumper.Dumper, instance: Enum):
        return dumper.represent_str(instance.name)

    def __str__(self):
        return str(self.value)


_PointedType = TypeVar("_PointedType")
_RootType = TypeVar("_RootType")


class Reference(Generic[_PointedType, _RootType]):
    def __init__(self, ref_str, referenced_obj=None):
        if not isinstance(ref_str, str):
            raise ValueError(
                f"References can only be constructed from strings, got {type(ref_str)} instead"
            )
        self._ref_str = ref_str
        self.referenced_obj = referenced_obj

    @classmethod
    def create(cls, root_type, obj):
        ref_str = root_type.get_reference_str(obj)
        return cls(ref_str, referenced_obj=obj)

    @property
    def id(self):  # noqa: A003
        rid = self._ref_str.split("/")[2].split("-")[0]
        return int(rid)

    @classmethod
    def yaml_representer(cls, dumper: yaml.dumper.Dumper, instance: "Reference"):
        return dumper.represent_str(instance._ref_str)

    def __repr__(self):
        if self._ref_str == "":
            return "<Invalid Reference>"
        return self._ref_str

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._ref_str == other._ref_str and self.referenced_obj == other.referenced_obj

    def is_valid(self):
        return self._ref_str != ""


def init_reference_yaml_classes(_: Type[yaml.Loader], dumper: Type[yaml.Dumper]):
    dumper.add_representer(Reference, Reference.yaml_representer)


def _field_is_default(field, value):
    field_metadata = field.metadata
    if field_metadata is None:
        return False
    is_optional = field_metadata.get("optional", False)
    if not is_optional:
        return False
    factory = field_metadata["default_value"]
    return value == factory()


class YamlLoader(Loader):
    pass


class YamlDumper(Dumper):
    def __init__(self, *args, **kwargs):
        # By default we emit an explicit document start (---) to make LLVM YAML parser happy
        if kwargs.get("explicit_start") is None:
            kwargs["explicit_start"] = True
        super().__init__(*args, **kwargs)

    def increase_indent(self, flow=False, _indentless=False):
        """Improves indentation"""
        return super().increase_indent(flow, False)

    def represent_str(self, data: str) -> yaml.ScalarNode:
        """Ensures literals starting with ? or : are quoted to make LLVM YAML parser happy"""
        node = super().represent_str(data)
        if data.startswith("?") or data.startswith(":"):
            node.style = '"'
        return node

    def ignore_aliases(self, data):
        return True


class DiffYamlLoader(YamlLoader):
    pass


class DiffYamlDumper(YamlDumper):
    pass


class TypedList(MutableSequence):
    def __init__(self, base_class: type):
        self._data: List[Any] = []
        self._base_class = base_class

    def __setitem__(self, idx, obj):
        if not isinstance(obj, self._base_class):
            raise ValueError(
                f"Cannot insert object, must be of type {self._base_class.__name__} (or subclass)"
            )
        self._data[idx] = obj

    def insert(self, index: int, obj):
        if not isinstance(obj, self._base_class):
            raise ValueError(
                f"Cannot insert object, must be of type {self._base_class.__name__} (or subclass)"
            )
        self._data.insert(index, obj)

    @classmethod
    def yaml_representer(cls, dumper: YamlDumper, instance) -> yaml.Node:
        return dumper.represent_list(instance._data)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self._data[idx]
        else:
            for d in self._data:
                if d.key() == str(idx):
                    return d
            raise IndexError("Index out of range")

    def __delitem__(self, idx):
        del self._data[idx]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._base_class == other._base_class and self._data == other._data
        else:
            return self._data == other


YamlDumper.add_representer(TypedList, TypedList.yaml_representer)


def typedlist_factory(base_class: type) -> Callable[[], TypedList]:
    def factory():
        return TypedList(base_class)

    return factory


def enum_value_to_index(enum_value: Enum):
    """Converts an enum value to its index"""
    return list(enum_value.__class__.__members__).index(enum_value.value)


# TODO: once tuple_tree_generator becomes kind-aware, this will no longer be necessary
def force_constructor_kwarg(base_class: type, kwarg_name, kwarg_value):
    """Monkeypatches the __init__ method so that the given kwarg is forced to the given value.
    If the argument was not provided it is set, if it was provided and it was different from the
    forced value a ValueError is raised.
    """
    assert isinstance(base_class, type)

    original_init = base_class.__init__  # type: ignore

    def init_forcing_value(self, *args, **kwargs):
        if kwarg_name not in kwargs:
            kwargs[kwarg_name] = kwarg_value
        else:
            if kwargs[kwarg_name] != kwarg_value:
                raise ValueError(f"Invalid value provided for {kwarg_name}")

        original_init(self, *args, **kwargs)

    base_class.__init__ = init_forcing_value  # type: ignore


# NOTE: remove after upgrade to python 3.10 (which introduces kw_only in dataclasses)
def force_kw_only(base_class: type):
    """Monkeypatches the __init__ method so that only kwargs are passed, raises ValueError if args
    contains values
    """
    assert isinstance(base_class, type)

    original_init = base_class.__init__  # type: ignore

    def init_check_kw_only(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            raise ValueError(f"Non-keyword arguments passed to constructor: {args}")
        original_init(self, *args, **kwargs)

    base_class.__init__ = init_check_kw_only  # type: ignore


def construct_type(type_info: dict, data):
    # if data not `str` or `dict` then its already constructed so we just return it.
    if not isinstance(data, (str, dict)):
        return data

    if type_info["ctor"] == "enum":
        return type_info["possibleValues"](data)
    elif type_info["ctor"] == "native":
        return type_info["type"](data)
    elif type_info["ctor"] == "parse":
        return type_info["type"](data)
    elif type_info["ctor"] == "class":
        return type_info["type"].from_dict(**data)
    else:
        raise RuntimeError(f"Unknown ctor: {type_info['ctor']}")


class DiffSet:
    def __init__(self, changes):
        self.changes = changes

    @classmethod
    def from_dict(cls, **kwargs):
        changes = []
        for diff in kwargs["Changes"]:
            changes.append(Diff(diff["Path"], diff.get("Add"), diff.get("Remove")))
        return DiffSet(changes)

    @classmethod
    def yaml_constructor(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return cls.from_dict(**mapping)

    @classmethod
    def yaml_representer(cls, dumper: YamlDumper, instance) -> yaml.Node:
        return dumper.represent_dict({"Changes": instance.changes})

    def __repr__(self):
        return repr({"Changes": self.changes})

    def __str__(self):
        return str({"Changes": self.changes})


class Diff:
    def __init__(self, path, add, remove):
        self.Path = path
        self.Add = add
        self.Remove = remove

    @classmethod
    def yaml_representer(cls, dumper: YamlDumper, instance):
        return dumper.represent_dict(instance._to_dict())

    def _to_dict(self):
        d = {"Path": self.Path}
        if self.Add:
            d["Add"] = self.Add
        if self.Remove:
            d["Remove"] = self.Remove
        return d

    def __repr__(self):
        return repr(self._to_dict())

    def __str__(self):
        return str(self._to_dict())

    def is_valid(self):
        return self.Add is not None or self.Remove is not None


YamlDumper.add_representer(Diff, Diff.yaml_representer)


def _parse_diff(diffs: DiffSet, get_type_info: Callable) -> DiffSet:
    new_changes = []
    for diff in diffs.changes:
        info = get_type_info(diff.Path)
        if diff.Add and diff.Remove:
            new_changes.append(
                Diff(diff.Path, construct_type(info, diff.Add), construct_type(info, diff.Remove))
            )
        elif diff.Remove:
            new_changes.append(Diff(diff.Path, None, construct_type(info, diff.Remove)))
        elif diff.Add:
            new_changes.append(Diff(diff.Path, construct_type(info, diff.Add), None))
    return DiffSet(new_changes)


def _get_type_info(path: str, root: StructBase, type_hints: dict) -> StructBase:
    path_parts = path.split("/")
    if len(path_parts) > 1:
        path_parts = path_parts[1:]

    root_hint = type_hints.get(root)
    if not root_hint:
        raise RuntimeError(f"Couldn't find root hint for: {root}")

    component = path_parts[0]
    composite_type = None
    if "::" in component:
        parts = component.split("::", 2)
        component = parts[1]
        composite_type = parts[0]

    if composite_type and "Type" in composite_type:
        if "Prototype" in root_hint:
            prototype_class = root_hint["Prototype"]["type"]
            prototype_class_children = getattr(prototype_class, "_children")
            composite_type_class = prototype_class_children[composite_type]
        elif "Kind" in root_hint:
            kind = root_hint["Kind"]["possibleValues"][composite_type]
            kind_name = getattr(kind, "name")
            root_children = getattr(root, "_children")
            composite_type_class = root_children[kind_name]

        composite_type_hint = type_hints[composite_type_class]
        root_hint.update(composite_type_hint)

    type_info = root_hint[component]

    if len(path_parts) == 1:
        return type_info

    if (
        type_info["isArray"]
        and type_info["ctor"] == "parse"
        and "parseKey" in type_info["type"].__dict__
    ):
        key = path_parts[1]
        key_parsed = type_info["type"].parseKey(key)
        if "Kind" in key_parsed:
            kind = key_parsed["Kind"]
            specialized_type = type_info["type"]._children.get(kind)
            return _get_type_info("/".join(path_parts[1:]), specialized_type, type_hints)
        return _get_type_info("/".join(path_parts[1:]), type_info["type"], type_hints)

    part = 2 if type_info["isArray"] else 1
    return _get_type_info("/".join(path_parts[part:]), type_info["type"], type_hints)


def _get_element_by_path_array(path: list, obj: StructBase) -> StructBase:
    component = path[0]
    if isinstance(obj, TypedList):
        for elem in obj:
            obj_fields = fields(elem)
            if hasattr(elem, "keyed") and elem.key() == component:
                if len(path) == 1:
                    return elem
                else:
                    return _get_element_by_path_array(path[1:], elem)
        raise RuntimeError(f"Couldn't find element: {elem} in obj: {obj}")

    else:
        if "::" in component:
            parts = component.split("::", 2)
            component = parts[1]
            kind = getattr(obj, "Kind", None)
            if not kind or kind.name != parts[0]:
                raise RuntimeError(f"Kind: {kind} doesn't exist or is not in {parts[0]}")
        obj_fields = fields(obj)
        for field in obj_fields:
            if path[0] == field.name or component == field.name:
                if len(path) == 1:
                    return getattr(obj, field.name)
                else:
                    return _get_element_by_path_array(path[1:], getattr(obj, field.name))
        else:
            raise RuntimeError(f"Couldn't find field: {obj_fields} in obj: {obj}")


def _get_element_by_path(path: str, obj: StructBase) -> StructBase:
    if path == "/":
        return obj
    return _get_element_by_path_array(path.split("/")[1:], obj)


def _get_parent_element_by_path(path: str, obj: StructBase) -> StructBase:
    parent_path = path.split("/")[1:-1]
    if not parent_path:
        return obj
    if len(parent_path) > 1:
        return _get_element_by_path(f"/{'/'.join(parent_path)}", obj)
    for f in fields(obj):
        if f.name == parent_path[0]:
            return getattr(obj, f.name)
    return obj


def _can_compare(info_object: dict) -> bool:
    if info_object["type"].__name__ == "str" and info_object["isArray"]:
        return False

    if info_object["type"].__name__ in ("str", "Reference", "Identifier", "MetaAddress"):
        return True

    return False


def _make_diff(
    obj_old: StructBase, obj_new: StructBase, type_hints: dict, root_type: StructBase
) -> DiffSet:
    root_type_info = {
        "type": root_type,
        "ctor": "class",
        "isArray": False,
        "optional": False,
        "isAbstract": False,
    }
    return DiffSet(_make_diff_subtree(obj_old, obj_new, "", type_hints, root_type_info, False))


def _make_diff_subtree(
    obj_old,
    obj_new,
    prefix: str,
    type_hints: dict,
    type_info: dict,
    in_array: bool,
) -> list:
    result = []
    if type(obj_old) is not type(obj_new):
        return []

    upcast = False
    if type_info["isAbstract"]:
        derived_type_children = getattr(type_info["type"], "_children")
        derived_class = derived_type_children.get(type(obj_old).__name__)
        base_info_object = type_hints[type_info["type"]]

        info_object = {}
        if derived_class:
            derived_class_object = type_hints[derived_class]
            info_object.update(base_info_object)
            info_object.update(derived_class_object)
            upcast = True
        else:
            info_object = base_info_object
    else:
        info_object = type_hints.get(type_info["type"], {})

    if type_info["isArray"] and not in_array:
        if hasattr(type_info["type"], "key"):
            map_old = {e.key(): e for e in obj_old}
            map_new = {e.key(): e for e in obj_new}
            common_keys = set()
            for key, value in map_old.items():
                if map_new.get(key):
                    common_keys.add(key)
                    result.extend(
                        _make_diff_subtree(
                            value, map_new.get(key), f"{prefix}/{key}", type_hints, type_info, True
                        )
                    )
                else:
                    result.append(Diff(prefix, None, value))
            for key, value in map_new.items():
                if key not in common_keys:
                    result.append(Diff(prefix, value, None))
        else:
            array_old = deepcopy(obj_old)
            array_new = deepcopy(obj_new)
            while len(array_old) > 0:
                head = array_old.pop(0)
                try:
                    new_idx = array_new.index(head)
                    del array_new[new_idx]
                except ValueError:
                    result.append(Diff(prefix, None, head))
            for e in array_new:
                result.append(Diff(prefix, e, None))

    else:
        for key in info_object:
            new_key = f"{obj_old.Kind}::{key}" if upcast else key
            key_prefix = f"{prefix}/{new_key}"
            if (
                not getattr(obj_old, key)
                and getattr(obj_new, key)
                and not info_object[key]["isArray"]
            ):
                result.append(Diff(f"{prefix}/{key}", getattr(obj_new, key), None))
            elif (
                getattr(obj_old, key)
                and not getattr(obj_new, key)
                and not info_object[key]["isArray"]
            ):
                result.append(Diff(f"{prefix}/{key}", None, getattr(obj_old, key)))
            elif (
                not getattr(obj_old, key)
                and not getattr(obj_new, key)
                and not info_object[key]["isArray"]
            ):
                # No changes, do nothing
                pass
            # flake8: noqa: SIM114
            elif info_object[key]["ctor"] == "native" and not info_object[key]["isArray"]:
                if getattr(obj_old, key) != getattr(obj_new, key):
                    result.append(Diff(key_prefix, getattr(obj_new, key), getattr(obj_old, key)))
            elif _can_compare(info_object[key]):
                if getattr(obj_old, key) != getattr(obj_new, key):
                    result.append(Diff(key_prefix, getattr(obj_new, key), getattr(obj_old, key)))
            else:
                result.extend(
                    _make_diff_subtree(
                        getattr(obj_old, key),
                        getattr(obj_new, key),
                        key_prefix,
                        type_hints,
                        info_object[key],
                        False,
                    )
                )
    return result


def _validate_diff(obj: StructBase, diffs: DiffSet, get_type_info: Callable) -> bool:
    for diff in diffs.changes:
        target = _get_element_by_path(diff.Path, obj)
        info = get_type_info(diff.Path)
        if info["isArray"]:
            if diff.Remove:
                if isinstance(target, TypedList):
                    array_target = [e for e in target if e == diff.Remove]
                    if not array_target:
                        return False
                else:
                    return False
            if diff.Add and isinstance(target, TypedList):
                array_target = [e for e in target if e == diff.Add]
                if array_target:
                    return False
        else:
            if diff.Remove and str(target) != str(diff.Remove):
                return False
            if diff.Add and diff.Remove and not target:
                return False
    return True


def _apply_diff(
    obj: StructBase, diffs: DiffSet, validate_diff: Callable
) -> tuple[bool, StructBase | None]:
    if not validate_diff(obj, diffs):
        return False, None
    new_obj = deepcopy(obj)
    for diff in diffs.changes:
        target = _get_element_by_path(diff.Path, new_obj)
        if isinstance(target, TypedList):
            if diff.Remove:
                for i, t in enumerate(target):
                    if t == diff.Remove:
                        idx = i
                        break
                del target[idx]
            if diff.Add:
                target.append(diff.Add)
        else:
            parent = _get_parent_element_by_path(diff.Path, new_obj)
            element = diff.Path.split("/")[::-1][0]
            if "::" in element:
                element = element.split("::", 2)[1]
            if diff.Remove:
                setattr(parent, element, None)
            if diff.Add and diff.Add != "":
                setattr(parent, element, diff.Add)

    return True, new_obj
