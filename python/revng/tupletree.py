#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
from collections.abc import MutableSequence
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
                isinstance(field, Reference) and not field.is_valid()
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
        # Prevent setting undefined attributes
        if self.__dataclass_fields__.get(key) is None:
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
        return self._data[idx]

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
