#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import abc
import sys
from collections.abc import Mapping, MutableSequence
from copy import deepcopy
from dataclasses import dataclass, fields
from enum import Enum, EnumType
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Generic, List, NotRequired
from typing import Optional, Sequence, Tuple, Type, TypeAlias, TypedDict, TypeVar, Union, get_args
from typing import get_origin, get_type_hints

import yaml

try:
    from yaml import CSafeDumper as Dumper
    from yaml import CSafeLoader as Loader
except ImportError:
    sys.stderr.write("Warning: using the slow pure-python YAML loader and dumper!\n")
    from yaml import SafeDumper as Dumper  # type: ignore
    from yaml import SafeLoader as Loader  # type: ignore

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

no_default = object()

dataclass_kwargs = {}
if sys.version_info >= (3, 10, 0):
    dataclass_kwargs["kw_only"] = True


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


DataclassT = TypeVar("DataclassT", bound="DataclassInstance")
EnumT = TypeVar("EnumT", bound=Enum)
T = TypeVar("T")


class TypeMetadata(TypedDict):
    type: Any  # noqa: A003
    possible_values: NotRequired[EnumT]  # type: ignore
    ctor: str
    optional: bool
    is_array: bool
    is_abstract: bool
    external: bool


TypesMetadata: TypeAlias = Dict[Type[Any], Dict[str, TypeMetadata]]
GetTypeMetadata: TypeAlias = Callable[[str], TypeMetadata]


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
            if origin in (list, TypedList):
                assert len(args) == 1
                underlying_type = args[0]

                if not isinstance(field_value, list):
                    raise ValueError(
                        f"Expected list for field {field_name} of {cls.__name__},"
                        + f"got {type(field_value)}"
                    )

                if origin is list:
                    instances = []
                else:
                    instances = TypedList(args[0])

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
    def yaml_representer(
        cls: Type[DataclassT], dumper: YamlDumper, instance: DataclassT
    ) -> yaml.Node:
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


StructBaseT = TypeVar("StructBaseT", bound=StructBase)


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
    def yaml_representer(cls: Type[EnumT], dumper: YamlDumper, instance: EnumT) -> yaml.Node:
        return dumper.represent_str(instance.name)

    def __str__(self):
        return str(self.value)


_PointedType = TypeVar("_PointedType")  # noqa: N808
_RootType = TypeVar("_RootType")  # noqa: N808


class Reference(Generic[_PointedType, _RootType]):
    def __init__(self, ref_str: str):
        if not isinstance(ref_str, str):
            raise ValueError(
                f"References can only be constructed from strings, got {type(ref_str)} instead"
            )
        self._ref_str = ref_str
        # To be filled later
        self._accessor: Optional[Callable[[str], StructBase]] = None

    def get(self) -> StructBase:
        assert self.is_valid(), "Reference is not valid"
        assert self._accessor is not None
        return self._accessor(self._ref_str)

    @property
    def id(self):  # noqa: A003
        return self._ref_str.split("/")[-1]

    @classmethod
    def yaml_representer(cls, dumper: yaml.dumper.Dumper, instance: "Reference"):
        return dumper.represent_str(instance._ref_str)

    def __repr__(self):
        if self._ref_str == "":
            return "<Invalid Reference>"
        return f"<Reference pointing to {self._ref_str}>"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._ref_str == other._ref_str

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


class TypedListDescriptor:
    def __init__(self, base_class):
        self.base_class = base_class
        self.name: str = ""

    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __get__(self, obj, objtype=None):
        val = getattr(obj, self.name, None)
        if val is not None:
            return val
        return TypedList(self.base_class)

    def __set__(self, obj, value):
        if value is self:
            setattr(obj, self.name, TypedList(self.base_class))
            return

        if isinstance(value, list):
            setattr(obj, self.name, TypedList(self.base_class, value))
        elif isinstance(value, TypedList):
            setattr(obj, self.name, value)
        else:
            raise ValueError(f"Error setting {value} as attribute")


class TypedList(Generic[T], MutableSequence, Mapping):
    def __init__(self, base_class: Type[T], list_: Sequence[T] = []):
        self._data: List[T] = []
        self._base_class = base_class
        if list_ != []:
            self.extend(list_)

    def __setitem__(self, idx: int, obj: T):  # type: ignore
        if not isinstance(obj, self._base_class):
            raise ValueError(
                f"Cannot insert object, must be of type {self._base_class.__name__} (or subclass)"
            )
        self._data[idx] = obj

    def insert(self, index: int, obj: T):
        if not isinstance(obj, self._base_class):
            raise ValueError(
                f"Cannot insert object, must be of type {self._base_class.__name__} (or subclass)"
            )
        self._data.insert(index, obj)

    @classmethod
    def yaml_representer(cls, dumper: YamlDumper, instance) -> yaml.Node:
        return dumper.represent_list(instance._data)

    def __getitem__(self, idx):
        """Fetching objects from this class works both as a normal python list
        and as a dictionary if the `base_class` is `keyed`.
        ```
        key = "..."
        example_list = TypedList(ExampleClass)
        example_list[0]                  # Normal list access
        example_list[ExampleClass(key)]  # Dictionary access with a keyable object
        example_list[key]                # Dictionary access with raw key
        ```
        """
        if isinstance(idx, (int, slice)):
            return self._data[idx]
        if getattr(self._base_class, "keyed", False):
            if isinstance(idx, str):
                key = idx
            else:
                key = idx.to_string()

            for d in self._data:
                if d.key() == key:
                    return d
            raise KeyError(idx)
        raise TypeError(f"Cannot index TypedList with {type(idx)}")

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

    def __contains__(self, item):
        if isinstance(item, self._base_class):
            return MutableSequence.__contains__(self, item)
        if getattr(self._base_class, "keyed", False):
            if isinstance(item, str):
                key = item
            else:
                key = item.to_string()
            return Mapping.__contains__(self, key)
        return False

    def __iter__(self):
        return self._data.__iter__()


YamlDumper.add_representer(TypedList, TypedList.yaml_representer)


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


def construct_type(type_metadata: TypeMetadata, data):
    """
    Build the type from the data.
    """
    # if data not `str` or `dict` then its already constructed so we just return it.
    if not isinstance(data, (str, dict)):
        return data

    if type_metadata["ctor"] == "enum":
        return type_metadata["possible_values"](data)  # type: ignore
    elif type_metadata["ctor"] == "native":
        return type_metadata["type"](data)
    elif type_metadata["ctor"] == "parse":
        # We need to check if dict, because otherwise it fails when parsing the
        # composite types
        if isinstance(data, dict):
            return type_metadata["type"].from_dict(**data)
        return type_metadata["type"](data)
    elif type_metadata["ctor"] == "class":
        return type_metadata["type"].from_dict(**data)
    else:
        raise RuntimeError(f"Unknown ctor: {type_metadata['ctor']}")


class Diff:
    def __init__(self, path: str, add, remove):
        self.Path = path
        self.Add = add
        self.Remove = remove

    @classmethod
    def yaml_representer(cls, dumper: YamlDumper, instance):
        return dumper.represent_dict(instance._to_dict())

    def _to_dict(self):
        d = {"Path": self.Path}
        if self.Add is not None:
            d["Add"] = self.Add
        if self.Remove is not None:
            d["Remove"] = self.Remove
        return d

    def __repr__(self):
        return repr(self._to_dict())

    def __str__(self):
        return str(self._to_dict())

    def is_valid(self):
        return self.Add is not None or self.Remove is not None


YamlDumper.add_representer(Diff, Diff.yaml_representer)


class DiffSet(abc.ABC, Generic[StructBaseT]):
    def __init__(self, changes: List[Diff]):
        self.Changes: List[Diff] = changes

    @staticmethod
    @abc.abstractmethod
    def _get_root() -> StructBase:
        ...

    @staticmethod
    @abc.abstractmethod
    def _get_types_metadata() -> TypesMetadata:
        ...

    @classmethod
    def make(cls, obj_old: StructBaseT, obj_new: StructBaseT) -> "DiffSet":
        return cls(_make_diff(obj_old, obj_new, cls._get_types_metadata(), cls._get_root()))

    @classmethod
    def _get_type_info(cls, path: str) -> TypeMetadata:
        return _get_type_info(path, cls._get_root(), cls._get_types_metadata())

    def validate(self, obj: StructBaseT) -> bool:
        return _validate_diff(obj, self, self._get_type_info)

    def apply(self, obj: StructBaseT):
        return _apply_diff(obj, self)

    @classmethod
    def from_dict(cls, dict_: dict) -> "DiffSet":
        changes = []
        for diff in dict_["Changes"]:
            changes.append(Diff(diff["Path"], diff.get("Add"), diff.get("Remove")))
        return cls(changes)

    @classmethod
    def yaml_constructor(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return cls.from_dict(**mapping)

    @classmethod
    def yaml_representer(cls, dumper: YamlDumper, instance) -> yaml.Node:
        return dumper.represent_dict({"Changes": instance.Changes})

    def __repr__(self):
        return repr({"Changes": self.Changes})

    def __str__(self):
        return str({"Changes": self.Changes})


def _parse_diff(diffs: DiffSet, get_type_metadata: GetTypeMetadata) -> DiffSet:
    """
    When we load the diff we also parse it so that we end up with objects. This comes in handy
    when we work further with the diff (apply) and we don't have to parse the diff again.
    """
    new_changes = []
    for diff in diffs.Changes:
        info = get_type_metadata(diff.Path)
        if diff.Add and diff.Remove:
            new_changes.append(
                Diff(diff.Path, construct_type(info, diff.Add), construct_type(info, diff.Remove))
            )
        elif diff.Remove:
            new_changes.append(Diff(diff.Path, None, construct_type(info, diff.Remove)))
        elif diff.Add:
            new_changes.append(Diff(diff.Path, construct_type(info, diff.Add), None))
    return diffs.__class__(new_changes)


def _get_type_info(
    path: Union[str, List[str]], root, types_metadata: TypesMetadata
) -> TypeMetadata:
    """
    Descend into the root (starting from Binary) recursively until we reach the
    the end node.
    """
    if isinstance(path, str):
        path_parts = path.split("/")[1:]
    else:
        path_parts = path

    root_hint = types_metadata.get(root)
    if not root_hint:
        raise RuntimeError(f"Couldn't find root hint for: {root}")

    component = path_parts[0]
    if "::" in component:
        # We have an upcastable type, the root_hint needs to be merged with the
        # actual subclass
        subtype_name, component = component.split("::", 1)
        subtype = root._children[subtype_name]
        root_hint = {**root_hint, **types_metadata[subtype]}

    type_info = root_hint[component]

    if len(path_parts) == 1:
        return type_info

    # If component is composite we need to find the specialized type
    if (
        type_info["is_array"]
        and type_info["ctor"] == "parse"
        and "parse_key" in type_info["type"].__dict__
    ):
        key = path_parts[1]
        key_parsed = type_info["type"].parse_key(key)
        kind = key_parsed["Kind"]
        specialized_type = type_info["type"]._children.get(kind)
        return _get_type_info("/".join(path_parts[1:]), specialized_type, types_metadata)

    part = 2 if type_info["is_array"] else 1
    return _get_type_info(path_parts[part:], type_info["type"], types_metadata)


def _get_element_by_path_array(path: list, obj: StructBase) -> StructBase:
    """
    Descend recursively into the path until we find the the `obj`.
    """
    component = path[0]
    if isinstance(obj, TypedList):  # noqa: R506
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
            parts = component.split("::", 1)
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
    """
    A wrapper around `_get_element_by_path_array` that splits the path into
    an array and returns the value from the recursive function.
    """
    if path == "/":
        return obj
    return _get_element_by_path_array(path.split("/")[1:], obj)


def _get_parent_element_by_path(path: str, obj: StructBase) -> StructBase:
    """
    Call `_get_element_by_path`, but return the parent of the `obj`.
    """
    parent_path = path.split("/")[1:-1]
    if not parent_path:
        return obj
    if len(parent_path) > 1:
        return _get_element_by_path(f"/{'/'.join(parent_path)}", obj)
    for f in fields(obj):
        if f.name == parent_path[0]:
            return getattr(obj, f.name)
    return obj


def _make_diff(
    obj_old: StructBase, obj_new: StructBase, types_metadata: TypesMetadata, root_type: StructBase
) -> List[Diff]:
    """
    A wrapper around `_make_diff_subtree` that returns a `DiffSet` object. This
    function allows the wrapped function to be calling itself recursively.
    """
    root_type_info: TypeMetadata = {
        "type": root_type,
        "ctor": "class",
        "is_array": False,
        "optional": False,
        "is_abstract": False,
        "external": False,
    }
    return _make_diff_subtree(obj_old, obj_new, "", types_metadata, root_type_info, False)


def _make_diff_subtree(
    obj_old,
    obj_new,
    prefix: str,
    type_metadata: TypesMetadata,
    type_info: TypeMetadata,
    in_array: bool,
) -> List[Diff]:
    """
    Generate the list of `Diff`s by traversing the `type_info` which is
    initially set to `Binary`. We explore all the nodes in the `Binary` and at
    each step we compare the values from `obj_old` and `obj_new`, if they
    differ we add to the `result`.
    """
    result = []
    assert type(obj_old) is type(obj_new)

    # If we have an abstract class, we need the `_children` to fetch the type
    # hints from and update the info object.
    upcast = False
    if type_info["is_abstract"]:
        derived_type_children = type_info["type"]._children
        derived_class = derived_type_children.get(type(obj_old).__name__)
        base_info_object = type_metadata[type_info["type"]]

        info_object = {}
        if derived_class:
            derived_class_object = type_metadata[derived_class]
            info_object.update(base_info_object)
            info_object.update(derived_class_object)
            upcast = True
        else:
            info_object = base_info_object
    else:
        info_object = type_metadata.get(type_info["type"], {})

    if type_info["is_array"] and not in_array:
        # If we have `key` attribute we use that to compare the values
        if getattr(type_info["type"], "keyed", False):
            map_old = {e.key(): e for e in obj_old}
            map_new = {e.key(): e for e in obj_new}
            common_keys = set()
            for key, value in map_old.items():
                if map_new.get(key):
                    common_keys.add(key)
                    result.extend(
                        _make_diff_subtree(
                            value,
                            map_new.get(key),
                            f"{prefix}/{key}",
                            type_metadata,
                            type_info,
                            True,
                        )
                    )
                else:
                    result.append(Diff(prefix, None, value))
            for key, value in map_new.items():
                if key not in common_keys:
                    result.append(Diff(prefix, value, None))
        else:
            # Otherwise compare the items at indexes
            array_old = list(obj_old)
            array_new = list(obj_new)
            while len(array_old) > 0:
                head = array_old.pop(0)
                if head in array_new:
                    new_idx = array_new.index(head)
                    del array_new[new_idx]
                else:
                    result.append(Diff(prefix, None, head))
            for e in array_new:
                result.append(Diff(prefix, e, None))

    else:
        # If we don't have an array then we loop over in attributes
        for key in info_object:
            new_key = f"{obj_old.Kind}::{key}" if upcast else key
            key_prefix = f"{prefix}/{new_key}"
            obj_old_key = getattr(obj_old, key)
            obj_new_key = getattr(obj_new, key)

            if obj_old_key is None and obj_new_key is not None:
                result.append(Diff(f"{prefix}/{key}", obj_new_key, None))
            elif obj_old_key is not None and obj_new_key is None:
                result.append(Diff(f"{prefix}/{key}", None, obj_old_key))
            elif obj_old_key is None and obj_new_key is None:
                # No changes, do nothing
                pass
            elif not info_object[key]["is_array"]:
                if obj_old_key != obj_new_key:
                    result.append(Diff(key_prefix, obj_new_key, obj_old_key))
            else:
                result.extend(
                    _make_diff_subtree(
                        obj_old_key,
                        obj_new_key,
                        key_prefix,
                        type_metadata,
                        info_object[key],
                        False,
                    )
                )
    return result


def _validate_diff(obj: StructBase, diffs: DiffSet, get_type_metadata: GetTypeMetadata) -> bool:
    """
    Check that the values from the `diffs` are present if we want to delete
    them or if the are not present if we want to add them.
    """
    for diff in diffs.Changes:
        target = _get_element_by_path(diff.Path, obj)
        info = get_type_metadata(diff.Path)
        if info["is_array"]:
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


def _apply_diff(obj: StructBase, diffs: DiffSet) -> Tuple[bool, Optional[StructBase]]:
    """
    Validate the diff and then proceed to remove/add from the `diffs`.
    """
    if not diffs.validate(obj):
        return False, None
    new_obj = deepcopy(obj)
    for diff in diffs.Changes:
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
                element = element.split("::", 1)[1]
            if diff.Remove:
                setattr(parent, element, None)
            if diff.Add and diff.Add != "":
                setattr(parent, element, diff.Add)

    return True, new_obj


VisitorEntry = Tuple[str, Any]
VisitorEntryGenerator = Generator[VisitorEntry, None, None]


class _FieldVisitor(Generic[StructBaseT]):
    def __init__(self, types_metadata: TypesMetadata, root):
        self.types_metadata = types_metadata
        self.root = root

    def visit_object(
        self, path: str, type_info: Dict[str, TypeMetadata], obj, obj_is_abstract=False
    ) -> VisitorEntryGenerator:
        if obj_is_abstract:
            new_path_prefix = f"{path}/{obj.Kind}::"
        else:
            new_path_prefix = f"{path}/"

        for key, info in type_info.items():
            new_path = f"{new_path_prefix}{key}"
            new_obj = getattr(obj, key)
            is_abc = info["ctor"] == "parse" and info["is_abstract"]
            is_class = info["ctor"] == "class" and not info["external"]
            is_complex = is_class or is_abc

            if is_complex:
                new_type_info = self.types_metadata[info["type"]]

                if is_abc:
                    derived_type_children = info["type"]._children
                    derived_class = derived_type_children.get(type(new_obj).__name__)

                    if derived_class:
                        new_type_info = {**new_type_info, **self.types_metadata[derived_class]}

            if info["is_array"]:
                for element in new_obj:
                    if is_complex:
                        element_path = f"{new_path}/{element.key()}"
                        yield (element_path, element)
                        yield from self.visit_object(element_path, new_type_info, element, is_abc)
                    else:
                        yield (new_path, element)
            else:
                yield (new_path, new_obj)
                if new_obj is None:
                    continue
                if is_complex:
                    yield from self.visit_object(new_path, new_type_info, new_obj, is_abc)

    def visit(self, obj) -> VisitorEntryGenerator:
        yield ("/", obj)
        yield from self.visit_object("", self.types_metadata[self.root], obj)
