#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

from abc import ABC, ABCMeta
from collections.abc import Buffer
from enum import Enum, EnumMeta, auto, unique
from typing import Any, Dict, Iterable, Mapping, Optional, TypeVar, Union, cast

import yaml

from revng.pypeline.analysis import Analysis
from revng.pypeline.container import Configuration, Container
from revng.pypeline.model import Model, ModelPath, ModelPathSet, ReadOnlyModel
from revng.pypeline.object import Kind, ObjectID, ObjectSet
from revng.pypeline.storage.file_provider import FileProvider
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.task import PipeObjectDependencies, TaskArgument, TaskArgumentAccess

Value = Union[str, int]
T = TypeVar("T")


def mandatory(arg: Optional[T]) -> T:
    assert arg is not None
    return arg


def only(value: list[Any]) -> Any:
    assert len(value) == 1
    return value[0]


class KindEnumMeta(ABCMeta, EnumMeta):
    """A metaclass that combines ABCMeta and EnumMeta to allow
    abstract methods in an Enum."""


@unique
class MyKind(Kind, Enum, metaclass=KindEnumMeta):
    ROOT = cast(Kind, auto())
    CHILD = cast(Kind, auto())
    GRANDCHILD = cast(Kind, auto())
    CHILD2 = cast(Kind, auto())

    @classmethod
    def kinds(cls) -> list[Kind]:
        return list(cast(Iterable[Kind], cls))

    def parent(self) -> Kind | None:
        # Small perf optimization: this could be moved outside
        hierarchy: dict[Kind, Kind | None] = {
            self.ROOT: None,
            self.CHILD: self.ROOT,
            self.GRANDCHILD: self.CHILD,
            self.CHILD2: self.ROOT,
        }
        return hierarchy[self]

    def __hash__(self) -> int:
        return hash(self.name)

    def serialize(self) -> str:
        return self.name

    @classmethod
    def deserialize(cls, obj: str) -> Kind:
        return MyKind[obj]

    def byte_size(self) -> int:
        return 0 if self == self.ROOT else 6


class MyObjectID(ObjectID):
    """An object ID that is a sequence of strings."""

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
        return MyObjectID(MyKind.ROOT)

    def parent(self) -> Optional[ObjectID]:
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

    def to_bytes(self) -> bytes:
        if self._kind == MyKind.ROOT:
            return b""

        def pad_null(string: str) -> bytes:
            bytes_ = string.encode()
            assert len(bytes_) <= 6
            return b"\x00" * (6 - len(bytes_)) + bytes_

        components = [pad_null(c) for c in self._components]
        if self._kind == MyKind.CHILD:
            return b"\x01" + components[0]
        elif self._kind == MyKind.GRANDCHILD:
            return b"\x01" + components[0] + b"\x03" + components[1]
        elif self._kind == MyKind.CHILD2:
            return b"\x02" + components[0]

        raise ValueError

    @classmethod
    def from_bytes(cls, bytes_: bytes) -> ObjectID:
        if len(bytes_) == 0:
            return MyObjectID(MyKind.ROOT)

        def parse_string(bytes2: bytes) -> str:
            string_bytes = b""
            for index, byte in enumerate(bytes2):
                if byte != 0:
                    string_bytes = bytes2[index:]
                    break
            return string_bytes.decode()

        if len(bytes_) == 7:
            if bytes_[0:1] == b"\x01":
                return MyObjectID(MyKind.CHILD, parse_string(bytes_[1:]))
            elif bytes_[0:1] == b"\x02":
                return MyObjectID(MyKind.CHILD2, parse_string(bytes_[1:]))
        elif len(bytes_) == 14 and bytes_[0:1] == b"\x01" and bytes_[7:8] == b"\x03":
            return MyObjectID(
                MyKind.GRANDCHILD, parse_string(bytes_[1:7]), parse_string(bytes_[8:])
            )

        raise ValueError


class DictContainer(Container, ABC):
    def __init__(self):
        self._object_list: ObjectSet = ObjectSet(self.kind)

    def clone(self, objects: Optional[ObjectSet]) -> DictContainer:
        # DictContainer is an abstract class that doesn't have a specific kind,
        # So to create a new instance, we get the current "subclass"
        # and instantiate it.
        result = self.__class__()
        if objects is not None:
            assert objects.issubset(self._object_list)
            result._object_list = objects  # pylint: disable=protected-access
        else:
            result._object_list = self._object_list  # pylint: disable=protected-access
        return result

    def merge(self, other: Container) -> None:
        # pylint: disable=protected-access
        self._object_list.update(cast(DictContainer, other)._object_list)

    def objects(self) -> ObjectSet:
        return ObjectSet(self._object_list.kind, set(self._object_list.objects))

    def erase_objects(self, objects: ObjectSet):
        self._object_list -= objects

    def clear(self):
        self._object_list = ObjectSet(self.kind)

    def verify(self) -> bool:
        return True

    def add_object(self, new_object: ObjectID):
        self._object_list.add(new_object)

    def __repr__(self):
        return f"DictContainer({self._object_list!r})"

    def __str__(self):
        return repr(self)

    @classmethod
    def mime_type(cls) -> str:
        return "text"

    def deserialize(self, data: Mapping[ObjectID, Buffer]) -> None:
        for oid, _ in data.items():
            self.add_object(oid)

    def serialize(self, objects: Optional[ObjectSet] = None) -> dict[ObjectID, Buffer]:
        if objects is None:
            return dict.fromkeys(self._object_list, b"")
        return dict.fromkeys(objects.objects, b"")


class RootDictContainer(DictContainer):
    kind = MyKind.ROOT


class ChildDictContainer(DictContainer):
    kind = MyKind.CHILD


class DictModel(Model):
    def __init__(self):
        self._data: Dict[ModelPath, Value] = {}

    def __contains__(self, path: ModelPath) -> bool:
        return path in self._data

    def __getitem__(self, path: ModelPath) -> Value:
        return self._data[path]

    def __setitem__(self, path: ModelPath, new_value: Value):
        self._data[path] = new_value

    def __delitem__(self, path: ModelPath):
        if path in self._data:
            del self._data[path]
        else:
            raise KeyError(f"ModelPath {path} not found in the model.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictModel):
            return False
        if set(self._data.keys()) != set(other._data.keys()):
            return False
        return all(self._data[key] == other._data[key] for key in self._data)

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> list[tuple[ModelPath, Value]]:
        return list(self._data.items())

    def keys(self) -> list[ModelPath]:
        return list(self._data.keys())

    def values(self) -> list[Value]:
        return list(self._data.values())

    def diff(self, other: DictModel) -> ModelPathSet:
        diff: ModelPathSet = set()
        for key, value in self.items():
            if key not in other or other[key] != value:
                diff.add(key)
        for key, value in other.items():
            if key not in self:
                diff.add(key)
        return diff

    def clone(self) -> DictModel:
        result = DictModel()
        result._data = dict(self._data)  # pylint: disable=protected-access
        return result

    def children(self, obj: ObjectID, kind: Kind) -> set[ObjectID]:
        if obj.kind() == MyKind.ROOT:
            if kind == MyKind.CHILD:
                return {
                    MyObjectID(MyKind.CHILD, "one"),
                    MyObjectID(MyKind.CHILD, "two"),
                    MyObjectID(MyKind.CHILD, "three"),
                }
            elif kind == MyKind.ROOT:
                return {MyObjectID(MyKind.ROOT)}

        raise NotImplementedError()

    @classmethod
    def mime_type(cls) -> str:
        return "application/x-yaml"

    @classmethod
    def model_name(cls) -> str:
        return "model.yml"

    def serialize(self):
        return yaml.safe_dump(self._data).encode()

    @classmethod
    def deserialize(cls, data: bytes):
        result = DictModel()
        result._data = yaml.safe_load(data)
        return result

    def __repr__(self):
        return f"DictModel({self._data!r})"

    def __str__(self):
        return repr(self)


class InPlacePipe(Pipe):
    """Modifies the input container in place."""

    @classmethod
    def signature(cls) -> tuple[TaskArgument, ...]:
        return (
            TaskArgument(
                "arg",
                ChildDictContainer,
                TaskArgumentAccess.READ_WRITE,
                help_text="the input container the pipe will modify",
            ),
        )

    def run(
        self,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        # Nothing to do
        return [[]]


class SameKindPipe(Pipe):
    @classmethod
    def signature(cls) -> tuple[TaskArgument, ...]:
        return (
            TaskArgument(
                "source",
                ChildDictContainer,
                TaskArgumentAccess.READ,
                help_text="the source container",
            ),
            TaskArgument(
                "destination",
                ChildDictContainer,
                TaskArgumentAccess.WRITE,
                help_text="the destination container",
            ),
        )

    def run(
        self,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        input_container: ChildDictContainer = cast(ChildDictContainer, containers[0])
        output_container: ChildDictContainer = cast(ChildDictContainer, containers[1])

        for obj in input_container.objects().objects:
            output_container.add_object(obj)

        return [[], []]


class ToHigherKindPipe(Pipe):
    """Take the root object from the input container and adds all
    its children to the output container."""

    @classmethod
    def signature(cls) -> tuple[TaskArgument, ...]:
        return (
            TaskArgument(
                "source",
                RootDictContainer,
                TaskArgumentAccess.READ,
                help_text="the source container",
            ),
            TaskArgument(
                "destination",
                ChildDictContainer,
                TaskArgumentAccess.WRITE,
                help_text="the destination container",
            ),
        )

    def run(
        self,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        input_container: RootDictContainer = cast(RootDictContainer, containers[0])
        input_kind = RootDictContainer.kind

        # Ensure we have the root object in input
        assert input_container.objects() == ObjectSet(input_kind, {MyObjectID.root()}), (
            f"Expected input container to contain only the root object, got: "
            f"{input_container.objects()}"
        )

        # Add all the children of the root object in output
        output_container: ChildDictContainer = cast(ChildDictContainer, containers[1])
        output_kind = output_container.kind
        root_object = MyObjectID.root()
        for obj in model.children(root_object, output_kind).objects:
            output_container.add_object(obj)

        return [[], []]


class ToLowerKindPipe(Pipe):
    @classmethod
    def signature(cls) -> tuple[TaskArgument, ...]:
        return (
            TaskArgument(
                "source",
                ChildDictContainer,
                TaskArgumentAccess.READ,
                help_text="the source container",
            ),
            TaskArgument(
                "destination",
                RootDictContainer,
                TaskArgumentAccess.WRITE,
                help_text="the destination container",
            ),
        )

    def run(
        self,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        input_container: ChildDictContainer = cast(ChildDictContainer, containers[0])
        input_kind = input_container.kind
        root_object = MyObjectID.root()

        # Ensure we have all the object we need in input
        assert input_container.objects() == model.children(root_object, input_kind)

        output_container: RootDictContainer = cast(RootDictContainer, containers[1])

        # Add to the output the root object
        output_container.add_object(MyObjectID.root())

        return [[], []]


class GeneratorPipe(Pipe):
    @classmethod
    def signature(cls) -> tuple[TaskArgument, ...]:
        return (
            TaskArgument(
                "arg",
                ChildDictContainer,
                TaskArgumentAccess.WRITE,
                help_text="the output container",
            ),
        )

    def __init__(
        self,
        static_configuration: str = "",
        name: str | None = None,
    ):
        super().__init__(
            name=name,
            static_configuration=static_configuration,
        )

    def run(
        self,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        dependencies: PipeObjectDependencies = []
        model = model.downcast()
        assert isinstance(model, DictModel), f"Model must be a DictModel got: {model!r}"
        container = cast(ChildDictContainer, containers[0])

        for objects in outgoing:
            for obj in objects:
                container.add_object(obj)
                dependencies.append((obj, "/one"))
        return [dependencies]


class NullAnalysis(Analysis):
    """An analysis that does nothing and returns an empty list of invalidations."""

    @classmethod
    def signature(cls) -> tuple[type[Container], ...]:
        return (ChildDictContainer,)

    def run(
        self,
        model: Model,
        containers: list[Container],
        incoming: list[ObjectSet],
        configuration: str,
    ):
        # This analysis does nothing
        pass


class PurgeOneAnalysis(Analysis):
    """An analysis that invalidates everything."""

    @classmethod
    def signature(cls) -> tuple[type[Container], ...]:
        return (ChildDictContainer,)

    def __init__(self, name: str):
        super().__init__(name)
        self.what_to_purge: list[ModelPath] = [
            "/one",
            "/test/test",
        ]

    def run(
        self,
        model: Model,
        containers: list[Container],
        incoming: list[ObjectSet],
        configuration: str,
    ):
        assert isinstance(model, DictModel)
        for purge_path in self.what_to_purge:
            if purge_path in model:
                del model[purge_path]


class PurgeAllAnalysis(Analysis):
    """An analysis that invalidates everything."""

    @classmethod
    def signature(cls) -> tuple[type[Container], ...]:
        return (ChildDictContainer,)

    def run(
        self,
        model: Model,
        containers: list[Container],
        incoming: list[ObjectSet],
        configuration: str,
    ):
        assert isinstance(model, DictModel)
        keys = model.keys()
        for key in keys:
            del model[key]


class AddStuffAnalysis(Analysis):
    """An analysis that invalidates everything."""

    @classmethod
    def signature(cls) -> tuple[type[Container], ...]:
        return (ChildDictContainer,)

    def __init__(self, name: str):
        super().__init__(name)
        self.what_to_add: list[ModelPath] = [
            "/one",
            "/test/test",
            "/test/hello",
        ]

    def run(
        self,
        model: Model,
        containers: list[Container],
        incoming: list[ObjectSet],
        configuration: str,
    ):
        assert isinstance(model, DictModel)
        for add_path in self.what_to_add:
            if add_path not in model:
                model[add_path] = f"wooo {add_path}"
