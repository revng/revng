#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import json
from collections.abc import Buffer, Mapping
from dataclasses import dataclass
from typing import Annotated, Dict, Generator, Tuple, Type

from .object import Kind, ObjectID, ObjectSet
from .utils.cabc import ABC, abstractmethod
from .utils.registry import get_singleton

ContainerID = Annotated[
    str,
    """
    This is used to identify a container in a savepoint.
    This is the name of the container.
    """,
]

ConfigurationId = Annotated[
    str,
    """
    From a pipeline node, this represent the static and runtime configuration of
    this node, and all its dependencies. This is used to index the storage provider
    in SavePoints.
    """,
]

Configuration = Annotated[
    str,
    "The runtime-configuration of a pipe, most commonly it will be a json / yaml string that"
    "the pipe will parse and use to configure itself."
    "THIS IS DIFFERENT FROM ConfigurationId.",
]


@dataclass(slots=True, frozen=True)
class ContainerDeclaration:
    """
    A ContainerDeclaration represents a container when describing a pipeline.
    It has a name and a type.

    Not to be confused with a Container, which is an instance of the container_type.
    """

    name: str
    container_type: Type[Container]

    def instance(self) -> Container:
        return self.container_type()


InvalidationList = list[Tuple[ConfigurationId, ContainerDeclaration, ObjectID]]


def group_by_container(
    invalidation_list: InvalidationList,
) -> Generator[Tuple[ConfigurationId, ContainerDeclaration, ObjectSet], None, None]:
    if len(invalidation_list) == 0:
        return

    first_configuration_id, first_container, _ = invalidation_list[0]
    last = (first_configuration_id, first_container)

    def new_object_list():
        return ObjectSet(first_container.container_type.kind, set())

    object_list: ObjectSet = new_object_list()
    for configuration_id, container, obj in invalidation_list:
        if last != (configuration_id, container):
            # We changed container. Yield what we accumulated so far and prepare a new list
            yield (last[0], last[1], object_list)
            last = (configuration_id, container)
            object_list = new_object_list()

        # Record the current object
        object_list.add(obj)

    # Yield the last group
    yield (last[0], last[1], object_list)


class Container(ABC):
    """
    A Container contains objects of a certain kind.
    """

    kind: Kind

    def __init__(self):
        """
        This constructor just makes it explicit that a container should be
        able to be initialized without any arguments.
        """

    @abstractmethod
    def objects(self) -> ObjectSet:
        pass

    @abstractmethod
    def deserialize(self, data: Mapping[ObjectID, Buffer]) -> None:
        """
        Ingest data from a serialized format into this container.
        This is used to **add** cached objects to this container.
        """

    @abstractmethod
    def serialize(self, objects: ObjectSet) -> Mapping[ObjectID, Buffer]:
        """
        Dump objects from this container into a serialized format.
        """

    @classmethod
    @abstractmethod
    def mime_type(cls) -> str:
        """
        The mime type of the serialized format of this container.
        This is used to inform the storage provider about the type of data
        it will be storing.
        """

    @classmethod
    def is_text(cls) -> bool:
        """
        Returns if the serialized format of this container is just
        text (e.g. JSON, YAML, etc.) or binary.
        This is used to improve transmission performance by avoiding
        unnecessary encoding/decoding steps.
        """
        return cls.mime_type().startswith("text/") or cls.mime_type() in {
            "application/json",
            "application/xml",
            "application/x-yaml",
        }

    @abstractmethod
    def verify(self) -> bool:
        pass

    def contains_all(self, obj: ObjectSet) -> bool:
        """
        Check if an object set is fully contained in this container.
        This is a default implementation, the container can probably do it much
        more efficiently, so you are supposed to override this method.
        """
        assert (
            self.kind == obj.kind
        ), f"Container {self} has kind {self.kind}, but the object set has kind {obj.kind}."
        return self.objects().issubset(obj)


def load_container(
    container_type: type[Container],
    path: str,
) -> Container:
    """
    Load a container from a serialized format.
    This is used to **load** cached objects into this container.
    """
    container = container_type()
    obj_id_ty = get_singleton(ObjectID)  # type: ignore[type-abstract]
    with open(path, "r", encoding="utf-8") as f:
        container.deserialize(
            {obj_id_ty.deserialize(k): bytes.fromhex(v) for k, v in json.load(f).items()}
        )
    return container


def dump_container(
    container: Container,
    path: str,
) -> None:
    """
    Dump a container into a serialized format.
    This is used to **save** cached objects from this container.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                k.serialize(): bytes(v).hex()
                for k, v in container.serialize(container.objects()).items()
            },
            f,
            indent=4,
            sort_keys=True,
        )


ContainerSet = Annotated[
    Dict[ContainerDeclaration, Container],
    """A set of bindings between container declarations and container instances.""",
]
