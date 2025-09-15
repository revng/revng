#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
Placeholder pipebox file that will be filled with the new nanobind-based
implementations. The code inside is only needed to pass the basic tests
that the code performs on the pipebox.
"""

from revng.pypeline.object import ObjectID, Kind, ObjectSet
from revng.pypeline.container import Container
from revng.pypeline.model import Model

class MyKind(Kind):
    """Test kind represented as a string"""
    def __init__(self, name: str):
        assert name in {"root", "function", "bb"}
        self.name = name

    @classmethod
    def kinds(cls) -> list[Kind]:
        return [MyKind("root"), MyKind("function"), MyKind("bb")]

    def parent(self) -> Kind | None:
        return {
            "root": None,
            "function": MyKind("root"),
            "bb": MyKind("function"),
        }[self.name]

    def __hash__(self) -> int:
        return hash(self.name)

    def serialize(self) -> str:
        return self.name

    @classmethod
    def deserialize(cls, value) -> Kind:
        return MyKind(value)

class MyObjectID(ObjectID):
    # Make ObjectIDs immutable
    def __init__(self, kind: Kind, *components: str):
        # A simple check to maintain structural integrity.
        if kind.rank() != len(components):
            raise ValueError("Number of components must match the kind's rank.")
        assert isinstance(kind, MyKind), f"Expected kind to be a MyKind, got: {kind!r}"
        self._kind: MyKind = kind
        self._components = tuple(components)  # Store as tuple to ensure immutability

    def kind(self) -> Kind:
        return self._kind

    @classmethod
    def root(cls) -> ObjectID:
        return MyObjectID(MyKind("root"))

    def parent(self) -> ObjectID | None:
        parent_kind = self._kind.parent()
        if parent_kind is None:
            return None
        return MyObjectID(parent_kind, *self._components[:-1])

    # since we have singleton items, the comparisons and hashes
    # can just be the object ptr
    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return hash((self._kind, self._components))

    def serialize(self) -> str:
        return f"/{self._kind.name}/{'/'.join(self._components)}"

    @classmethod
    def deserialize(cls, obj: str) -> ObjectID:
        """Deserialize an object id of this class"""
        kind, *components = obj.strip("/").split("/")
        return cls(MyKind.deserialize(kind), *components)

class ListContainer(Container):
    def __init__(self):
        raise NotImplementedError()
    def objects(self) -> ObjectSet:
        raise NotImplementedError()
    def deserialize(self, data) -> None:
        raise NotImplementedError()
    def serialize(self, objects = None):
        raise NotImplementedError()
    @classmethod
    def mime_type(cls) -> str:
        raise NotImplementedError()
    def verify(self) -> bool:
        raise NotImplementedError()

class RootContainer(ListContainer):
    kind = MyKind("root")

class FuncContainer(ListContainer):
    kind = MyKind("function")

class BbContainer(ListContainer):
    kind = MyKind("bb")

class MyModel(Model):
    def diff(self, other):
        raise NotImplementedError()
    def clone(self):
        raise NotImplementedError()
    def children(self, obj: ObjectID, kind: Kind) -> ObjectSet:
        raise NotImplementedError()
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()
    def serialize(self) -> bytes:
        raise NotImplementedError()
    def deserialize(self, data: bytes):
        raise NotImplementedError()
