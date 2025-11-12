#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import base64
import io
import json
import tarfile
from collections.abc import Buffer, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Annotated, Generator, Literal, Tuple, Type, final

from .object import Kind, ObjectID, ObjectSet
from .utils import bytes_to_string, is_mime_type_text, string_to_bytes
from .utils.cabc import ABC, abstractmethod
from .utils.registry import get_singleton

ContainerFormat = Literal["tar", "json"]

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
    def serialize(self, objects: ObjectSet | None = None) -> Mapping[ObjectID, Buffer]:
        """
        Dump objects from this container into a serialized format, dump all
        objects if none are specified.
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
        return is_mime_type_text(cls.mime_type())

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

    @classmethod
    @final
    def from_dict(cls, data: dict[str, bytes]) -> Container:
        # Feed the data in the container
        container = cls()
        obj_id_type = get_singleton(ObjectID)  # type: ignore[type-abstract]
        container.deserialize(
            {obj_id_type.deserialize(obj_id): content for obj_id, content in data.items()}
        )
        return container

    @classmethod
    @final
    def from_bytes(
        cls,
        data: bytes,
        container_format: ContainerFormat = "json",
    ) -> Container:
        """
        Load a container into a serialized format.
        """
        match container_format:
            case "json":
                return cls.from_dict(
                    {
                        obj_id: string_to_bytes(content, cls.is_text())
                        for obj_id, content in json.loads(data.decode("utf-8")).items()
                    }
                )
            case "tar":
                data_dict: dict[str, bytes] = {}
                with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
                    for member in tar.getmembers():
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        data_dict[member.name] = f.read()
                return cls.from_dict(data_dict)
        raise ValueError(f"Unknown container format: {container_format}")

    @classmethod
    @final
    def from_string(
        cls,
        data: str,
        container_format: ContainerFormat = "json",
    ) -> Container:
        """
        Load a container into a serialized format.
        """
        match container_format:
            case "json":
                return cls.from_bytes(data.encode("utf-8"), container_format="json")
            case "tar":
                return cls.from_bytes(base64.b64decode(data), container_format="tar")
        raise ValueError(f"Unknown container format: {container_format}")

    @classmethod
    def container_format_from_bytes(cls, data: bytes) -> ContainerFormat:
        is_tar = tarfile.is_tarfile(io.BytesIO(data))
        if is_tar:
            return "tar"
        elif data.startswith(b"{"):
            return "json"
        else:
            raise ValueError(f"Cannot detect the container format for {data[:20]!r}")

    @classmethod
    @final
    def from_file(
        cls,
        path_or_file: str | Path,
        container_format: ContainerFormat | Literal["auto"] = "auto",
    ) -> Container:
        """
        Load a container from a serialized format.
        This is used to **load** cached objects into this container.
        Container format "auto" will guess the format automatically.
        """
        with open(path_or_file, "rb") as f:
            data = f.read()
            if container_format == "auto":
                container_format = cls.container_format_from_bytes(data)
            return cls.from_bytes(data, container_format=container_format)

    @final
    def to_dict(self) -> dict[str, Buffer]:
        return {k.serialize(): v for k, v in self.serialize(self.objects()).items()}

    @final
    def to_bytes(self, container_format: ContainerFormat = "json") -> bytes:
        """
        Dump a container into a serialized format.
        """
        match container_format:
            case "json":
                with StringIO() as string_stream:
                    data = {
                        key: bytes_to_string(buffer, self.is_text())
                        for key, buffer in self.to_dict().items()
                    }
                    json.dump(
                        data,
                        string_stream,
                        indent=4,
                        sort_keys=True,
                    )
                    return string_stream.getvalue().encode("utf-8")
            case "tar":
                with io.BytesIO() as byte_stream:
                    with tarfile.open(fileobj=byte_stream, mode="w") as tar:
                        for obj_id, buffer in self.to_dict().items():
                            info = tarfile.TarInfo(name=obj_id)
                            data_bytes = bytes(buffer)
                            info.size = len(data_bytes)
                            tar.addfile(tarinfo=info, fileobj=io.BytesIO(data_bytes))
                    return byte_stream.getvalue()
        raise ValueError(f"Unknown container format: {container_format}")

    @final
    def to_string(self, container_format: ContainerFormat = "json") -> str:
        """
        Dump a container into a serialized format.
        """
        match container_format:
            case "json":
                return self.to_bytes(container_format="json").decode("utf-8")
            case "tar":
                return base64.b64encode(self.to_bytes(container_format="tar")).decode("utf-8")
        raise ValueError(f"Unknown container format: {container_format}")

    @final
    def to_file(
        self, path: str | Path, container_format: ContainerFormat | Literal["auto"] = "auto"
    ) -> None:
        """
        Dump a container into a serialized format.
        This is used to **save** cached objects from this container.
        ContainerFormat "auto" will guess the format from the path.
        """
        if container_format == "auto":
            if str(path).endswith(".json"):
                container_format = "json"
            elif str(path).endswith(".tar"):
                container_format = "tar"
            else:
                raise ValueError(
                    f"Cannot determine the container_format for {path}.",
                )

        with open(path, "wb") as f:
            f.write(self.to_bytes(container_format=container_format))


ContainerSet = Annotated[
    dict[ContainerDeclaration, Container],
    """A set of bindings between container declarations and container instances.""",
]
