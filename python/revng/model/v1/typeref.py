import yaml
from pydantic import BaseModel, Extra, Field

class Typeref(BaseModel):
    class Config:
        extra = Extra.forbid

    __root__: str = Field(
        ...,
    )

    @staticmethod
    def create(revng_type):
        typedef_str = Typeref.get_typeref_str(revng_type)
        return Typeref(__root__=typedef_str)

    @staticmethod
    def get_typeref_str(revng_type):
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


def typeref_yaml_representer(dumper: yaml.dumper.Dumper, instance: Typeref):
    return dumper.represent_str(repr(instance))


yaml.add_representer(
    Typeref,
    typeref_yaml_representer,
)

__all__ = [
    "Typeref"
]
