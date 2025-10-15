#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Collection, Iterable, Mapping

from revng.pypeline.container import ConfigurationId, ContainerID
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies

SavepointID = Annotated[
    int,
    """
    An integer that represents a savepoint, we want this to be an integer so we
    can assign them doing a DFS traversal of the pipeline, which in turns allows
    to efficiently represent a subtree of savepoints as a continuous range of
    integers to avoid storing the dependencies multiple times.
    """,
]


@dataclass(frozen=True, slots=True)
class ContainerLocation:
    savepoint_id: SavepointID
    container_id: ContainerID
    configuration_id: ConfigurationId


InvalidatedObjects = dict[ContainerLocation, set[ObjectID]]


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


class StorageProvider(ABC):
    """This is the general interface for something that caches containers.
    This can be in memory, on disk, in a database, etc.
    This is a singleton and there should never be more than one instance of it.
    """

    @abstractmethod
    def has(
        self,
        location: ContainerLocation,
        keys: Collection[ObjectID],
    ) -> Iterable[ObjectID]:
        """
        Get the available objects from the storage.
        If the object is not found, it will not be included in the result.
        """

    @abstractmethod
    def get(
        self,
        location: ContainerLocation,
        keys: Collection[ObjectID],
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
    def invalidate(self, invalidation_list: ModelPathSet) -> InvalidatedObjects:
        """
        Inform the storage that certain model paths are no longer valid.
        The storage should use the stored dependencies to determine which objects
        need to be invalidated.
        It returns the list of invalidated objects, in each savepoint.
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
