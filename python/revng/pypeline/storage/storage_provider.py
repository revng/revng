#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Buffer
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, AsyncContextManager, Collection, Iterable, Mapping
from urllib.parse import urlparse

from revng.pypeline.container import ConfigurationId, ContainerID
from revng.pypeline.model import Model, ModelPathSet
from revng.pypeline.object import ObjectID, ObjectSet
from revng.pypeline.storage.notification_queue import LOCAL_QUEUE
from revng.pypeline.task.pipe import ObjectDependencies, PipeCustomInvalidation
from revng.pypeline.utils.registry import get_registry

from .file_provider import FileProvider, FileRequest

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


InvalidatedObjects = dict[ContainerLocation, set[ObjectID]]


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    last_change: datetime
    version: str


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


@dataclass
class FileStorageEntry:
    # Name to use for the file in case it needs to be stored on disk
    name: str
    # Path of the file to store, this will avoid loading the file contents in
    # memory all at once
    path: Path | None = field(default=None)
    # Contents of the file to store
    contents: bytes | None = field(default=None)

    def __post_init__(self):
        assert int(self.path is not None) + int(self.contents is not None) == 1


@dataclass(frozen=True, slots=True)
class ObjectsToInvalidate:
    savepoint_range: SavePointsRange
    container_id: ContainerID
    configuration_id: ConfigurationId
    objects: ObjectSet


@dataclass
class SetModelResult:
    # The new epoch after changing the model
    epoch: int
    # The objects that have been pruned from storage, stored as a dictionary
    # that maps the container location to a list of objects
    invalidated_objects: InvalidatedObjects


@dataclass
class PipeDependencies:
    pipe_id: int
    savepoints_range: SavePointsRange
    configuration: ConfigurationId
    dependencies: ObjectDependencies
    custom_invalidation: PipeCustomInvalidation

    def empty(self) -> bool:
        return len(self.dependencies) == 0 and not self.has_custom_invalidation()

    def has_custom_invalidation(self):
        return not all(len(x) == 0 for x in self.custom_invalidation)


class StorageProviderFactory(ABC):
    """
    A possibly stateful factory for a specific storage provider.
    This abstraction is needed to optimize daemon, as the cloud version needs to
    be stateless and instantiate a new storage provider client each time, while
    the local version can reuse the same client.
    """

    @abstractmethod
    def __init__(self, url: str):
        """
        The cli will decide how to instantiate the factory using a url argument
        so this url has to contain all the information needed to instantiate the
        factory.
        """

    @classmethod
    @abstractmethod
    def scheme(cls) -> str:
        """
        The scheme that identifies this storage provider, i.e. the initial part
        of the URL before the "://".
        """

    @abstractmethod
    def get(
        self,
        base_directory: Path,
        project_id: ProjectID | None,
        token: str | None,
        cache_dir: str | None,
    ) -> AsyncContextManager[StorageProvider]:
        """
        Get a storage provider instance for the given project.
        This method has to receive all the arguments needed by all possible
        implementations, so it's common that the implementation will ignore some
        of them. The method returns an async context manager that yields the
        storage provider. This is done because we need to lock the storage provider
        based on the project ID because only one user at time can modify a project.
        """

    @abstractmethod
    def get_notification_websocket(self) -> str | None:
        """
        Get the URL of the notification websocket. If the function returns None
        then the internal websocket will be used.
        """


# TODO: find a more suitable name
def storage_provider_factory_factory(url: str) -> StorageProviderFactory:
    """Create a storage provider factory from an url."""
    scheme = urlparse(url).scheme
    factories = get_registry(StorageProviderFactory)  # type: ignore [type-abstract]
    for factory_type in factories.values():
        if factory_type.scheme() == scheme:
            return factory_type(url)
    available_schemes = ", ".join(factories.keys())
    raise ValueError(
        f"Unknown storage provider scheme {scheme}."
        f" The available schemes are: {available_schemes}."
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
    def add_objects(
        self,
        dependencies: list[PipeDependencies],
        objects: Mapping[ContainerLocation, Mapping[ObjectID, Buffer]],
    ) -> None:
        """
        Store dependencies and custom invalidation for an arbitrary number of
        pipes while storing objects to storage. This guarantees that the
        objects stored and their dependencies are stored atomically and avoids
        dangling dependencies.
        """

    @abstractmethod
    def get_epoch(self) -> int:
        """Get the epoch, i.e. the model version number."""

    @abstractmethod
    def get_model(self) -> tuple[Model, int]:
        """Get the model and the epoch."""

    @abstractmethod
    def set_model(
        self,
        new_model: Model,
        changed_paths: ModelPathSet,
        custom_invalidations: list[ObjectsToInvalidate],
    ) -> SetModelResult:
        """
        Inform the storage that the model has changed, with the list of changed
        model paths and object to explicitly invalidate (from custom
        invalidation). This function will compute the overall list of objects
        to invalidate and set the model. It will return the new epoch (which
        will be current epoch + 1 if the model changed, or current epoch if it
        didn't) and the exhaustive list of objects that have been deleted due
        to the changes in the model.
        This requires the new model and the list of changed paths separately so
        that the storage provider does not need to run operations (diff/apply)
        on the model.
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

    @abstractmethod
    def put_files_in_storage(self, files: list[FileStorageEntry]) -> list[str]:
        """
        Put a file in storage, which can be later retrieved with the
        `get_files_from_storage` method. This method returns a list of
        hashes of the submitted files.
        """

    @abstractmethod
    def get_files_from_storage(self, requests: list[FileRequest]) -> dict[str, bytes]:
        """
        Get a file from storage, given a list of requests. Returns a dictionary
        mapping the hash to the contents.
        """

    @abstractmethod
    def get_custom_invalidation_data(
        self, pipe_id: int, configuration_hash: str
    ) -> PipeCustomInvalidation:
        """
        Retrieve custom invalidation data previously stored by
        `add_custom_invalidation_data`.
        """

    @staticmethod
    def _send_local_invalidation(invalidated: InvalidatedObjects, epoch: int):
        """
        Send invalidation data to the local queue. Only to be used on local
        storage providers where notifications are not handled by the provider.
        """

        payload = {
            "type": "invalidation",
            "epoch": epoch,
            "invalidated": [
                {
                    "savepoint_id": location.savepoint_id,
                    "container_id": location.container_id,
                    "configuration": location.configuration_id,
                    "object_ids": [object_id.serialize() for object_id in object_ids],
                }
                for location, object_ids in invalidated.items()
            ],
        }
        LOCAL_QUEUE.send(json.dumps(payload).encode())


class StorageProviderFileProvider(FileProvider):
    def __init__(self, provider: StorageProvider):
        self._provider = provider

    def get_files(self, requests: list[FileRequest]) -> dict[str, bytes]:
        return self._provider.get_files_from_storage(requests)
