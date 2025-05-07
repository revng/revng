#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Type

import yaml


class MetaAddressType(Enum):
    Invalid = "Invalid"
    Generic32 = "Generic32"
    Generic64 = "Generic64"
    Code_x86 = "Code_x86"
    Code_x86_64 = "Code_x86_64"
    Code_mips = "Code_mips"
    Code_mipsel = "Code_mipsel"
    Code_arm = "Code_arm"
    Code_arm_thumb = "Code_arm_thumb"
    Code_aarch64 = "Code_aarch64"
    Code_systemz = "Code_systemz"

    @classmethod
    def yaml_representer(cls, dumper: yaml.dumper.Dumper, instance: Enum):
        return dumper.represent_str(instance.name)


@dataclass
class MetaAddress:
    Address: int = field(default=0)
    Type: MetaAddressType = field(default=MetaAddressType.Invalid)
    Epoch: int = field(default=0)
    AddressSpace: int = field(default=0)

    @classmethod
    def from_string(cls, s):
        match = cls.yaml_regexp.fullmatch(s)
        if match is None:
            raise ValueError(f"Could not parse {s} as a MetaAddress")

        address = match["Address"] or "0"
        meta_address_type = match["Type"] or "Invalid"
        epoch = match["Epoch"] or "0"
        address_space = match["AddressSpace"] or "0"

        return cls(
            Address=cls._parse_int(address),
            Type=MetaAddressType(meta_address_type),
            Epoch=cls._parse_int(epoch),
            AddressSpace=cls._parse_int(address_space),
        )

    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**kwargs)

    def is_default_epoch(self):
        return self.Epoch == 0

    def is_default_address_space(self):
        return self.AddressSpace == 0

    def is_invalid(self):
        return self.Type == MetaAddressType.Invalid

    def to_string(self) -> str:
        if self.Address == 0:
            addr = ""
        else:
            addr = hex(self.Address)

        components = [
            addr,
            self.Type.name,
        ]
        if not self.is_default_epoch():
            components.append(str(self.Epoch))
        if not self.is_default_address_space():
            components.append(str(self.AddressSpace))
        return ":".join(components)

    def __repr__(self) -> str:
        return self.to_string()

    @classmethod
    def yaml_representer(cls, dumper: yaml.dumper.Dumper, instance: "MetaAddress"):
        return dumper.represent_str(instance.to_string())

    @staticmethod
    def _parse_int(i):
        if isinstance(i, int):
            return i
        return int(i, base=0)

    yaml_regexp = re.compile(
        # Start of the string
        r"\A"
        # Address (can be empty for invalid MetaAddresses, ":Invalid")
        "(?P<Address>(0x[0-9a-fA-F]+)|)"
        # Type
        rf""":(?P<Type>{"|".join(v.name for v in MetaAddressType)})"""
        # Optional epoch
        rf"""(:(?P<Epoch>\d+))?"""
        # Optional address space
        rf"""(:(?P<AddressSpace>\d+))?"""
        # End of the string
        r"\Z"
    )


def init_metaaddress_yaml_classes(loader: Type[yaml.Loader], dumper: Type[yaml.Dumper]):
    def metaaddress_constructor(loader: yaml.Loader, node):
        string = loader.construct_scalar(node)
        return MetaAddress.from_string(string)

    dumper.add_representer(MetaAddress, MetaAddress.yaml_representer)
    dumper.add_representer(MetaAddressType, MetaAddressType.yaml_representer)
    loader.add_constructor("!MetaAddress", metaaddress_constructor)
    loader.add_implicit_resolver("!MetaAddress", MetaAddress.yaml_regexp, None)
