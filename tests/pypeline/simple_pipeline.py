#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# pylint: disable=too-many-positional-arguments

from __future__ import annotations

import sys
from abc import ABC
from typing import Any, Dict, Mapping, Optional, TypeVar, Union, cast

import yaml

from revng.pypeline.container import Configuration, Container
from revng.pypeline.model import Model, ModelPath, ModelPathSet, ReadOnlyModel
from revng.pypeline.object import ObjectID, ObjectSet
from revng.pypeline.pipeline import Analysis
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


class RootObjectID(ObjectID):
    __slots__ = ()

    def __eq__(self, other) -> bool:
        return True

    def __hash__(self) -> int:
        return 0

    def parent(self) -> Optional[ObjectID]:
        return None

    def serialize(self) -> str:
        return "/" + self.__class__.__name__ + "/"

    @classmethod
    def deserialize(cls, obj: str) -> ObjectID:
        assert obj.startswith("/")
        ty, *components = obj[1:].split("/")
        assert ty == cls.__name__
        return RootObjectID()


assert sys.getsizeof(RootObjectID()) / 8 == 4


class ChildObjectID(ObjectID, parent_kind=RootObjectID):
    __slots__ = ("identifier",)

    def __init__(self, identifier: str):
        self.identifier: str = identifier

    def __eq__(self, other) -> bool:
        return self.identifier == other.identifier

    def __hash__(self) -> int:
        return hash(self.identifier)

    def parent(self) -> Optional[ObjectID]:
        return RootObjectID()

    def serialize(self) -> str:
        return "/" + self.__class__.__name__ + "/" + self.identifier

    @classmethod
    def deserialize(cls, obj: str) -> ObjectID:
        assert obj.startswith("/")
        ty, *components = obj[1:].split("/")
        assert len(components) == 1
        assert ty == cls.__name__
        return ChildObjectID(components[0])


assert sys.getsizeof(ChildObjectID("")) / 8 == 5


class DictContainer(Container, ABC):
    def __init__(self):
        self._object_list = ObjectSet(self.kind)

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

    def deserialize(self, data: Mapping[ObjectID, bytes]) -> None:
        for oid, _ in data.items():
            self.add_object(oid)

    def serialize(self, objects: Optional[ObjectSet] = None) -> dict[ObjectID, bytes]:
        if objects is None:
            objects = self._object_list
        return dict.fromkeys(objects.objects, b"")


class RootDictContainer(DictContainer):
    kind = RootObjectID


class ChildDictContainer(DictContainer):
    kind = ChildObjectID


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
        return diff

    def clone(self) -> DictModel:
        result = DictModel()
        result._data = dict(self._data)  # pylint: disable=protected-access
        return result

    def children(self, obj: ObjectID, kind: type[ObjectID]) -> ObjectSet:
        if isinstance(obj, RootObjectID):
            if kind is ChildObjectID:
                return ObjectSet(
                    ChildObjectID,
                    {
                        ChildObjectID("one"),
                        ChildObjectID("two"),
                        ChildObjectID("three"),
                    },
                )
            elif kind is RootObjectID:
                return ObjectSet(RootObjectID, {RootObjectID()})

        raise NotImplementedError()

    @classmethod
    def is_text(cls) -> bool:
        return True  # We serialize already as json

    def serialize(self):
        return yaml.safe_dump(self._data).encode()

    def deserialize(self, data: bytes):
        self._data = yaml.safe_load(data)

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
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        # Nothing to do
        return []


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

        return []


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
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        input_container: RootDictContainer = cast(RootDictContainer, containers[0])
        input_kind = RootDictContainer.kind

        # Ensure we have the root object in input
        assert input_container.objects() == ObjectSet(input_kind, {RootObjectID()}), (
            f"Expected input container to contain only the root object, got: "
            f"{input_container.objects()}"
        )

        # Add all the children of the root object in output
        output_container: ChildDictContainer = cast(ChildDictContainer, containers[1])
        output_kind = output_container.kind
        root_object = RootObjectID()
        for obj in model.children(root_object, output_kind).objects:
            output_container.add_object(obj)

        return []


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
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
        input_container: ChildDictContainer = cast(ChildDictContainer, containers[0])
        input_kind = input_container.kind
        root_object = RootObjectID()

        # Ensure we have all the object we need in input
        assert input_container.objects() == model.children(root_object, input_kind)

        output_container: RootDictContainer = cast(RootDictContainer, containers[1])

        # Add to the output the root object
        output_container.add_object(RootObjectID())

        return []


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
                dependencies.append((0, obj, "/one"))
        return dependencies


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
            "/ciao/ciao",
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
            "/ciao/ciao",
            "/ciao/hello",
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
