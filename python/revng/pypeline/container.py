#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import base64
import io
import tarfile
from collections.abc import Buffer, Mapping
from dataclasses import dataclass
from enum import StrEnum
from io import StringIO
from pathlib import Path
from typing import Annotated, Generator, Tuple, Type, final

import yaml

from .compression import Compression
from .object import Kind, ObjectID, ObjectSet
from .utils import bytes_to_string, is_mime_type_text, string_to_bytes
from .utils.cabc import ABC, abstractmethod
from .utils.registry import get_registry, get_singleton


class ContainerFormat(StrEnum):
    """
    The serialization format of a container.

    AUTO is output-only: it emits a lone object raw (no YAML/base64 wrapping)
    and otherwise behaves like YAML. It is not a valid input format.
    """

    AUTO = "auto"
    TAR = "tar"
    YAML = "yaml"

    @classmethod
    def detect_format(cls, data: bytes) -> ContainerFormat:
        """Detect the container format from the given data."""
        # Try to detect the format from the data
        is_tar = tarfile.is_tarfile(io.BytesIO(data))
        if is_tar:
            return ContainerFormat.TAR
        # Try to decode as yaml
        try:
            yaml.safe_load(data.decode("utf-8"))
            return ContainerFormat.YAML
        except yaml.YAMLError:
            raise ValueError(f"Cannot detect the container format for {data[:20]!r}")


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


@dataclass(slots=True, frozen=True, order=True)
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

    def to_dict(self) -> dict:
        """Convert the data into a dictionary representation."""
        return {
            "name": self.name,
            "type": self.container_type.name,
        }


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

    name: str
    kind: Kind
    compression: str

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

    @abstractmethod
    def set_is_disposable(self):
        """
        Sets that the container's content is not going to be used shortly.
        This allows calls to `dispose_if_possible` to actually remove the content.
        """

    @abstractmethod
    def dispose_if_possible(self):
        """
        If `set_is_disposable` has been called, prune the container, otherwise do
        nothing.
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
        return obj.issubset(self.objects())

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
    def from_bytes(cls, data: bytes, container_format: ContainerFormat | None = None) -> Container:
        """
        Load a container into a serialized format. If container_format is None,
        the format will be detected automatically.
        """
        match container_format:
            case None:
                return cls.from_bytes(
                    data,
                    container_format=ContainerFormat.detect_format(data),
                )
            case ContainerFormat.YAML:
                return cls.from_dict(
                    {
                        obj_id: string_to_bytes(content, cls.is_text())
                        for obj_id, content in yaml.safe_load(data.decode("utf-8")).items()
                    }
                )
            case ContainerFormat.TAR:
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
        container_format: ContainerFormat = ContainerFormat.YAML,
    ) -> Container:
        """
        Load a container into a serialized format.
        """
        match container_format:
            case ContainerFormat.YAML:
                return cls.from_bytes(data.encode("utf-8"), container_format=container_format)
            case ContainerFormat.TAR:
                return cls.from_bytes(base64.b64decode(data), container_format=container_format)
        raise ValueError(f"Unknown container format: {container_format}")

    @classmethod
    @final
    def from_file(
        cls,
        path_or_file: str | Path,
        container_format: ContainerFormat | None = None,
    ) -> Container:
        """
        Load a container from a serialized format.
        This is used to **load** cached objects into this container.
        If container_format is None, the format will be detected automatically.
        """
        with open(path_or_file, "rb") as f:
            return cls.from_bytes(f.read(), container_format=container_format)

    @final
    def to_dict(self, objects: ObjectSet | None = None) -> dict[str, Buffer]:
        if objects is not None:
            assert self.contains_all(objects)
        else:
            objects = self.objects()
        return {k.serialize(): v for k, v in self.serialize(objects).items()}

    @final
    def to_bytes(
        self,
        objects: ObjectSet | None = None,
        container_format: ContainerFormat = ContainerFormat.YAML,
    ) -> bytes:
        """
        Dump a container into a serialized format.
        """
        match container_format:
            case ContainerFormat.YAML:
                with StringIO() as string_stream:
                    data = {
                        key: bytes_to_string(buffer, self.is_text())
                        for key, buffer in self.to_dict(objects).items()
                    }
                    yaml.dump(data, string_stream)
                    return string_stream.getvalue().encode("utf-8")
            case ContainerFormat.TAR:
                with io.BytesIO() as byte_stream:
                    with tarfile.open(fileobj=byte_stream, mode="w") as tar:
                        for obj_id, buffer in self.to_dict(objects).items():
                            info = tarfile.TarInfo(name=obj_id)
                            data_bytes = bytes(buffer)
                            info.size = len(data_bytes)
                            tar.addfile(tarinfo=info, fileobj=io.BytesIO(data_bytes))
                    return byte_stream.getvalue()
            case ContainerFormat.AUTO:
                # A lone object is emitted raw, without any YAML/base64 wrapping;
                # zero or multiple objects fall back to YAML.
                target = self.objects() if objects is None else objects
                if len(target) == 1:
                    (buffer,) = self.serialize(target).values()
                    return bytes(buffer)
                return self.to_bytes(objects, ContainerFormat.YAML)
        raise ValueError(f"Unknown container format: {container_format}")

    @final
    def to_string(
        self,
        objects: ObjectSet | None = None,
        container_format: ContainerFormat = ContainerFormat.YAML,
    ) -> str:
        """
        Dump a container into a serialized format.
        """
        match container_format:
            case ContainerFormat.YAML:
                return self.to_bytes(objects, ContainerFormat.YAML).decode("utf-8")
            case ContainerFormat.TAR:
                return base64.b64encode(self.to_bytes(objects, ContainerFormat.TAR)).decode("utf-8")
            case ContainerFormat.AUTO:
                # A lone textual object is returned raw; anything else is YAML.
                target = self.objects() if objects is None else objects
                if len(target) == 1 and self.is_text():
                    return self.to_bytes(objects, ContainerFormat.AUTO).decode("utf-8")
                return self.to_string(objects, ContainerFormat.YAML)
        raise ValueError(f"Unknown container format: {container_format}")

    @final
    def to_file(
        self,
        path: str | Path,
        objects: ObjectSet | None = None,
        container_format: ContainerFormat | None = None,
    ) -> None:
        """
        Dump a container into a serialized format.
        This is used to **save** cached objects from this container.
        If container_format is None, the format will be detected automatically.
        """
        if container_format is None:
            if str(path).endswith(".yaml") or str(path).endswith(".yml"):
                container_format = ContainerFormat.YAML
            elif str(path).endswith(".tar"):
                container_format = ContainerFormat.TAR
            else:
                raise ValueError(
                    f"Cannot determine the container_format for {path}.",
                )

        with open(path, "wb") as f:
            f.write(self.to_bytes(objects, container_format))

    @classmethod
    def get_compression(cls) -> Compression | None:
        if cls.compression == "none":
            return None
        elif ";" in cls.compression:
            type_, options = cls.compression.split(";", 1)
            return get_registry(Compression)[type_](options)  # type: ignore[type-abstract]
        else:
            return get_registry(Compression)[cls.compression]()  # type: ignore[type-abstract]

    @classmethod
    def type_dict(cls) -> dict:
        """Convert the data into a dictionary representation."""
        return {
            "class": cls.name,
            "mime_type": cls.mime_type(),
            "is_text": cls.is_text(),
            "kind": cls.kind.serialize(),
            "compression": cls.compression,
        }


ContainerSet = Annotated[
    dict[ContainerDeclaration, Container],
    """A set of bindings between container declarations and container instances.""",
]
