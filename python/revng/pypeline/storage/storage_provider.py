#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Iterable, Mapping

from revng.pypeline.container import ConfigurationId, ContainerID
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies
from revng.pypeline.utils.registry import get_registry

SavepointID = Annotated[
    int,
    """
    An integer that represents a savepoint, we want this to be an integer so we
    can assign them doing a DFS traversal of the pipeline, which in turns allows
    to efficiently represent a subtree of savepoints as a continuous range of
    integers to avoid storing the dependencies multiple times.
    """,
]

ProjectID = Annotated[
    str,
    """
    An unique identifier for a project.
    """,
]


@dataclass(frozen=True, slots=True)
class ContainerLocation:
    savepoint_id: SavepointID
    container_id: ContainerID
    configuration_id: ConfigurationId


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    last_change: datetime
    revng_version: str


@dataclass(frozen=True, slots=True)
class SavePointsRange:
    """A range of savepoints, with inclusive extremes.
    When the savepoints IDs are assigned in a DFS traversal of the pipeline,
    any subtree of savepoints can be represented as a continuous range of integers,
    avoiding the need to store the dependencies multiple times.
    The starts are assigned with a pre-order traversal, and the ends are assigned
    with a post-order traversal. The order is increase only when a savepoint is
    visited for the first time.
    """

    start: SavepointID
    """
    This is the smallest savepoint ID of the subtree of a node (root included).
    Thus, if the node is a SavePoint, this is the ID of the SavePoint itself.
    """

    end: SavepointID
    """This is the largest savepoint ID of the subtree of a node (root included).
    Thus, if the node is a SavePoint, this is start + the number of savepoints
    present in the subtree - 1 (minus the root itself)."""

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, int):
            return False
        return self.start <= item <= self.end

    def __len__(self) -> int:
        """The number of savepoints in the range."""
        return self.end - self.start + 1


class StorageProviderFactory(ABC):
    """
    A possibly stateful factory for a specific storage provider.
    This abstraction is needed to optimize daemon, as the cloud version needs to
    be stateless and instantiate a new storage provider client each time, while
    the local version can reuse the same client.
    """

    @abstractmethod
    def __init__(self, route: str):
        """
        The cli will decide how to instantiate the factory using an route argument
        so this route has to contain all the information needed to instantiate the
        factory.
        """

    @classmethod
    @abstractmethod
    def protocol(cls) -> str:
        """
        The protocol that identifies this storage provider.
        For example if the protocol of an implementation is `local`
        the code has to instantiate it when it receives an url in
        the form `local://<route>`.
        """

    @abstractmethod
    def get(
        self,
        project_id: ProjectID | None,
        token: str | None,
        cache_dir: str | None,
    ) -> StorageProvider:
        """
        Get a storage provider instance for the given project.
        This method has to receive all the arguments needed by all possible
        implementations, so it's common that the implementation will ignore some
        of them. The methods are not supposed to be thread-safe, and it's
        responsibility of the user to ensure that this method is not called
        concurrently.
        """


# WIP: find a more suitable name
def storage_provider_factory_factory(url: str) -> StorageProviderFactory:
    """Create a storage provider factory from an url."""
    protocol, route = url.split("://", 1)
    factories = get_registry(StorageProviderFactory)  # type: ignore [type-abstract]
    for factory_ty in factories.values():
        if factory_ty.protocol() == protocol:
            return factory_ty(route)
    available_protocols = ", ".join(factories.keys())
    raise ValueError(
        f"Unknown storage provider protocol {protocol}."
        f" The available protocols are: {available_protocols}."
    )


class StorageProvider(ABC):
    """This is the general interface for something that caches containers.
    This can be in memory, on disk, in a database, etc.
    This is a singleton and there should never be more than one instance of it.
    """

    @abstractmethod
    def has(
        self,
        location: ContainerLocation,
        keys: Iterable[ObjectID],
    ) -> Iterable[ObjectID]:
        """
        Get the available objects from the storage.
        If the object is not found, it will not be included in the result.
        """

    @abstractmethod
    def get(
        self,
        location: ContainerLocation,
        keys: Iterable[ObjectID],
    ) -> Mapping[ObjectID, bytes]:
        """
        For each objects, return bytes that the container can ingest to
        deserialize the object.
        All the object **have to** be present, as one should call `get_available`
        first to check it.
        """

    @abstractmethod
    def add_dependencies(
        self,
        savepoint_range: SavePointsRange,
        configuration_id: ConfigurationId,
        deps: ObjectDependencies,
    ) -> None:
        """
        Store the dependencies between the objects and the model paths.
        This is has to be called **BEFORE** `put`.
        It can, and probably will, contain duplicated dependencies from previous
        calls, but the storage should handle this gracefully.
        """

    @abstractmethod
    def put(
        self,
        location: ContainerLocation,
        values: Mapping[ObjectID, bytes],
    ) -> None:
        """
        Put a set of serialized objects into the storage.
        This has always to be called **AFTER** `add_dependencies`
        """

    @abstractmethod
    def invalidate(self, invalidation_list: ModelPathSet) -> None:
        """
        Inform the storage that certain model paths are no longer valid.
        The storage should use the stored dependencies to determine which objects
        need to be invalidated.
        """

    @abstractmethod
    def get_model(self) -> bytes:
        """
        Get the model
        """

    @abstractmethod
    def set_model(self, new_model: bytes):
        """
        Set the model
        """

    @abstractmethod
    def metadata(self) -> ProjectMetadata:
        """
        Fetch metadata about the current project
        """

    @abstractmethod
    def prune_objects(self):
        """
        Prunes all the objects (except metadata) from storage
        """
