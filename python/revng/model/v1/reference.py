#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import yaml
from pydantic import BaseModel, Extra, Field, PrivateAttr


class Reference(BaseModel):
    class Config:
        extra = Extra.forbid

    __root__: str = Field(
        ...,
    )
    _original_ref = PrivateAttr()

    def __init__(self, *, __root__):
        # Allow constructing references directly from revng types
        if not isinstance(__root__, str):
            self._original_ref = __root__
            __root__ = self.get_reference_str(__root__)
        super().__init__(__root__=__root__)

    @staticmethod
    def create(revng_type):
        typedef_str = Reference.get_reference_str(revng_type)
        return Reference(__root__=typedef_str)

    @staticmethod
    def get_reference_str(revng_type):
        # TODO: make this not-model specific
        if hasattr(revng_type, "Kind"):
            typename = str(revng_type.Kind)
        else:
            typename = type(revng_type).__name__
        id = revng_type.ID
        return f"/Types/{typename}-{id}"

    @property
    def id(self):
        _, _, id = self.__root__.rpartition("-")
        return int(id)

    def __repr__(self):
        return self.__root__


def reference_yaml_representer(dumper: yaml.dumper.Dumper, instance: Reference):
    return dumper.represent_str(repr(instance))


yaml.add_representer(
    Reference,
    reference_yaml_representer,
)

__all__ = ["Reference"]
