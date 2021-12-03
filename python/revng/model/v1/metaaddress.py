import re
from enum import Enum, auto
from typing import Optional

import yaml
from pydantic import BaseModel, Extra, Field, PrivateAttr


class MetaAddressType(Enum):
    Invalid = auto()
    Generic32 = auto()
    Generic64 = auto()
    Code_x86 = auto()
    Code_x86_64 = auto()
    Code_mips = auto()
    Code_mipsel = auto()
    Code_arm = auto()
    Code_arm_thumb = auto()
    Code_aarch64 = auto()
    Code_systemz = auto()


class MetaAddress(BaseModel):
    class Config:
        extra = Extra.forbid

    # Do not remove the leading underscore, otherwise pydantic will treat this as a property
    # and crash while trying to find an appropriate validator for it
    _yaml_regexp = re.compile(
        # Address (can be empty for invalid MetaAddresses, ":Invalid")
        "(?P<Address>(0x[0-9a-fA-F]+)|)"
        # Type
        rf""":(?P<Type>{"|".join(v.name for v in MetaAddressType)})"""
        # Optional epoch
        rf"""(:(?P<Epoch>\d+))?"""
        # Optional address space
        rf"""(:(?P<AddressSpace>\d+))?"""
    )

    __root__: str = Field(
        ...,
        regex=_yaml_regexp.pattern,
    )

    _Address: int = PrivateAttr()
    _Type: MetaAddressType = PrivateAttr()
    _Epoch: Optional[int] = PrivateAttr(default=0)
    _AddressSpace: Optional[int] = PrivateAttr(default=0)

    def __init__(self, **kwargs):
        assert ("__root__" in kwargs) ^ ("Address" in kwargs and "Type" in kwargs), (
            "MetaAddress can be constructed by providing it in string form using the __root__ kwarg "
            " or by explicitly providing Address, Type, and optional Epoch and AddressSpace"
        )

        if "__root__" in kwargs:
            kwargs = self._parse_string(kwargs["__root__"])

        self._Address = kwargs["Address"]
        self._Type = kwargs["Type"]
        self._Epoch = kwargs.get("Epoch", 0)
        self._AddressSpace = kwargs.get("AddressSpace", 0)
        super(MetaAddress, self).__init__(__root__=repr(self))

    @property
    def Address(self):
        return self._Address

    @Address.setter
    def Address(self, value):
        self._Address = value
        self._update_root()

    @property
    def Type(self):
        return self._Type

    @Type.setter
    def Type(self, value):
        self._Type = value
        self._update_root()

    @property
    def Epoch(self):
        return self._Epoch

    @Epoch.setter
    def Epoch(self, value):
        self._Epoch = value
        self._update_root()

    @property
    def AddressSpace(self):
        return self._AddressSpace

    @AddressSpace.setter
    def AddressSpace(self, value):
        self._AddressSpace = value
        self._update_root()

    def _update_root(self):
        self.__root__ = repr(self)

    @classmethod
    def _parse_string(cls, s: str):
        assert isinstance(s, str)

        match = cls._yaml_regexp.match(s)
        if match is None:
            raise ValueError(f"Could not parse {s} as a MetaAddress")

        address = match["Address"] or "0"
        meta_address_type = match["Type"]
        epoch = match["Epoch"] or "0"
        address_space = match["AddressSpace"] or "0"

        return {
            "Address": int(address, base=0),
            "Type": MetaAddressType[meta_address_type],
            "Epoch": int(epoch, base=0),
            "AddressSpace": int(address_space, base=0),
        }

    def is_default_epoch(self):
        return self.Epoch == 0

    def is_default_address_space(self):
        return self.AddressSpace == 0

    def is_invalid(self):
        return self._Type == MetaAddressType.Invalid

    def __eq__(self, other):
        if not isinstance(other, MetaAddress):
            return False

        return (
            self._Address == other._Address
            and self._Epoch == other._Epoch
            and self._Type == other._Type
            and self._AddressSpace == other._AddressSpace
        )

    def __hash__(self):
        return self._Address

    def __repr__(self):
        components = [
            hex(self._Address),
            self._Type.name,
        ]
        if not self.is_default_epoch():
            components.append(str(self._Epoch))
        if not self.is_default_address_space():
            components.append(str(self._AddressSpace))

        return ":".join(components)


def metaaddr_yaml_representer(dumper: yaml.dumper.Dumper, instance: MetaAddress):
    return dumper.represent_str(repr(instance))


yaml.add_representer(
    MetaAddress,
    metaaddr_yaml_representer,
)

__all__ = [
    "MetaAddress",
    "MetaAddressType",
]
