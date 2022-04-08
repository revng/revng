import sys
from dataclasses import dataclass, fields
from enum import Enum
from functools import lru_cache
from typing import Dict, Generic, Type, TypeVar, get_args, get_origin, get_type_hints

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    sys.stderr.write("Warning: using the slow pure-python YAML loader and dumper!\n")
    from yaml import Loader, Dumper

no_default = object()


def _create_instance(field_value, field_type):
    # First we check if the value is already of the required type
    if isinstance(field_value, field_type):
        return field_value
    elif isinstance(field_value, str) and issubclass(field_type, Enum):
        return field_type(field_value)
    elif isinstance(field_value, dict) and issubclass(field_type, StructBase):
        return field_type.from_dict(**field_value)
    elif isinstance(field_value, str) and hasattr(field_type, "from_string"):
        return field_type.from_string(field_value)
    else:
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
                        f"Expected list for field {field_name} of {cls.__name__}, got {type(field_value)}"
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
            if _field_is_optional(field) and (field_val is None or field_val == []):
                continue
            mapping_to_dump[field.name] = field_val

        return dumper.represent_mapping(f"!{cls.__name__}", mapping_to_dump)

    def __post_init__(self):
        # Before python 3.10 dataclasses had an annoying limitation regarding inheritance:
        # if the base class has a field with a default value, *all* the fields of the child classes
        # need to have a default value too, or dataclasses will raise a TypeError.
        # Since python 3.10 this limitation can be mostly bypassed by defining the base class
        # fields as kw_only, but we want to support older python versions.
        # Hence this workaround, inspired by https://stackoverflow.com/a/53085935
        for field in fields(self):
            if self.__getattribute__(field.name) is no_default:
                raise TypeError(f"__init__ missing 1 required argument: {field.name}")

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
        else:
            return super().from_dict(**kwargs)


class EnumBase(Enum):
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
    def id(self):
        _, _, id = self._ref_str.rpartition("-")
        return int(id)

    @classmethod
    def yaml_representer(cls, dumper: yaml.dumper.Dumper, instance: "Reference"):
        return dumper.represent_str(repr(instance))

    def __repr__(self):
        return self._ref_str


def init_reference_yaml_classes(loader: Type[yaml.Loader], dumper: Type[yaml.Dumper]):
    dumper.add_representer(Reference, Reference.yaml_representer)


def _field_is_optional(field):
    field_metadata = field.metadata
    if field_metadata is None:
        return False
    return field_metadata.get("optional", False)


class YamlLoader(Loader):
    pass


class YamlDumper(Dumper):
    def __init__(self, *args, **kwargs):
        # By default we emit an explicit document start (---) to make LLVM YAML parser happy
        if kwargs.get("explicit_start") is None:
            kwargs["explicit_start"] = True
        super(YamlDumper, self).__init__(*args, **kwargs)

    def increase_indent(self, flow=False, indentless=False):
        """Improves indentation"""
        return super(YamlDumper, self).increase_indent(flow, False)

    def represent_str(self, data: str) -> yaml.ScalarNode:
        """Ensures literals starting with ? or : are quoted to make LLVM YAML parser happy"""
        node = super().represent_str(data)
        if data.startswith("?") or data.startswith(":"):
            node.style = '"'
        return node

    def ignore_aliases(self, data):
        return True
