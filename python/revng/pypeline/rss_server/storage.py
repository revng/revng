#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import asyncio
import enum
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TypedDict


class LockRequest(enum.Enum):
    ARTIFACT = enum.auto()
    ANALYSIS = enum.auto()


class LockCheckType(enum.Enum):
    # No lock required, check will always return True
    NONE = enum.auto()
    # Any lock, as long as it's valid, will return True
    ANY = enum.auto()
    # Requires an Artifact/Analysis lock
    ARTIFACT = enum.auto()
    # Requires an Analysis lock
    ANALYSIS = enum.auto()


class CheckLockInvalid(Exception):
    """Exception throw by hold_lock if the lock is invalid"""


@dataclass(frozen=True, slots=True)
class InvalidatedObject:
    """A single invalidated object row returned after invalidation."""

    object_id: str
    container_id: str
    savepoint_id: int
    configuration_hash: str


class ProjectMetadataRow(TypedDict):
    version: str | None
    pipeline_description_hash: str | None
    last_fetch: float
    last_object_save: float
    last_model_save: float


@dataclass(frozen=True, slots=True)
class DependencyEntry:
    """A single dependency to store."""

    savepoint_id_start: int
    savepoint_id_end: int
    container_id: str
    configuration_hash: str
    object_id: bytes
    model_path: str


@dataclass(frozen=True, slots=True)
class AdditionalObjectEntry:
    savepoint_id_start: int
    savepoint_id_end: int
    container_id: str
    configuration_hash: str
    object_id: bytes


@dataclass(frozen=True, slots=True)
class CustomDependency:
    """A single custom dependency to store."""

    pipe_id: int
    configuration_hash: str
    argument_index: int
    object_id: bytes
    data: bytes


@dataclass(frozen=True, slots=True)
class CustomDependencyEntry:
    argument_index: int
    object_id: bytes
    data: bytes


@dataclass(frozen=True, slots=True)
class ObjectEntry:
    """A single object to store."""

    savepoint_id: int
    container_id: str
    configuration_hash: str
    object_id: bytes
    object_id_string: str
    content: bytes


@dataclass(frozen=True, slots=True)
class ModelSetResult:
    """Result of an invalidate-and-set-model operation."""

    new_epoch: int
    invalidated: list[InvalidatedObject]


class CheckMetadataResult(enum.Enum):
    OK = enum.auto()
    PIPELINE_DESCRIPTION_HASH_MISSING = enum.auto()
    PRUNE_DONE = enum.auto()


class RSSStorage(ABC):
    @staticmethod
    @abstractmethod
    def migrate(connection_string: str):
        """Apply migrations if needed"""

    @staticmethod
    @abstractmethod
    def background_tasks(connection_string: str) -> list[asyncio.Task]:
        """Create the background tasks needed for housekeeping"""

    @staticmethod
    @abstractmethod
    async def make(connection_string: str) -> RSSStorage:
        """Constructor method for the class"""

    @abstractmethod
    async def close(self):
        """Release any resources held by the storage backend."""

    @abstractmethod
    def get_project_storage(self, project_id: str) -> RSSProjectStorage:
        """Get the project storage for the specified project ID"""


class RSSProjectStorage(ABC):
    #  Locking primitives

    @abstractmethod
    async def make_lock(self, lock_type: LockRequest) -> str:
        """
        Request a new lock of `lock_type`. This function will wait until the
        lock is ready to be used.
        """

    @abstractmethod
    async def renew_lock(self, lock_id: str) -> bool:
        """
        Renew the lock's refresh timestamp.
        Returns False if the lock does not exist or is expired.
        """

    @abstractmethod
    async def release_lock(self, lock_id: str):
        """
        Release the lock. This will also trigger the DB to unlock as many new
        waiting locks as possible.
        """

    #  Metadata

    @abstractmethod
    async def get_metadata(self) -> ProjectMetadataRow:
        """Return the metadata of a project"""

    @abstractmethod
    async def create_project_if_missing(self):
        """Check if the project ID is present in the DB, if not add it"""

    @abstractmethod
    async def upgrade_project(
        self, version: str, pipeline_description_hash: str
    ) -> CheckMetadataResult:
        """
        Check that the version and has of the pipeline description match those
        stored on the DB. If not all the objects will be pruned.
        """

    @abstractmethod
    async def put_pipeline_description(self, hash_: str, content: bytes):
        """Store a pipeline description blob"""

    # Locked instance accessor

    @abstractmethod
    def get_locked(
        self, lock_id: str, initial_type: LockCheckType
    ) -> AbstractAsyncContextManager[RSSLockedProjectStorage]:
        """
        Get the LockedProjectStorage instance, the lock used will be held until
        the context manager is closed.
        """


class RSSLockedProjectStorage(ABC):
    #  Savepoints

    @abstractmethod
    async def has_objects(
        self,
        savepoint_id: int,
        container_id: str,
        configuration_hash: str,
        object_ids: list[bytes],
    ) -> list[bytes]:
        """
        Given a list of wanted `object_ids` return a list of the subset that is
        present in storage.
        """

    @abstractmethod
    async def get_objects(
        self,
        savepoint_id: int,
        container_id: str,
        configuration_hash: str,
        object_ids: list[bytes],
    ) -> list[tuple[bytes, bytes]]:
        """
        Given a list of `object_ids` that *need* to be present in the DB,
        retrieve them and return their contents.
        """

    @abstractmethod
    async def get_custom_invalidation_data(
        self,
        pipe_id: int,
        configuration_hash: str,
    ) -> list[CustomDependencyEntry]:
        """
        Given a pipe_id and configuration_hash, return all the entries in the
        DB that match.
        """

    @abstractmethod
    async def add_objects(
        self,
        dependencies: list[DependencyEntry],
        custom_dependencies: list[CustomDependency],
        objects: list[ObjectEntry],
    ):
        """
        Store a group of objects and their dependencies into the DB
        """

    @abstractmethod
    async def prune_objects(self):
        """
        Delete all objects, dependencies, and custom dependencies for a project.
        """

    #  Model

    @abstractmethod
    async def get_epoch(self) -> int:
        """Return the current epoch for the project"""

    @abstractmethod
    async def get_model(self) -> tuple[bytes | None, int]:
        """Return the model (in bytes) and the epoch"""

    @abstractmethod
    async def invalidate_and_set_model(
        self,
        model_paths: list[str],
        additional_objects: list[AdditionalObjectEntry],
        model_bytes: bytes,
    ) -> ModelSetResult:
        """
        Given a new model and the invalidation it caused (in the form of
        changed model paths and objects that need to be removed), do the
        following:
        1. Run invalidation so that any object related to the modified paths
           is dropped (in addition to the explicitly-specified
           `additional_objects`)
        2. Set the new model, increment the epoch by 1
        """

    #  File hashmap

    @abstractmethod
    async def put_file(self, hash_: str, content: bytes):
        """Store a file blob by its hash"""

    @abstractmethod
    async def get_files(self, hashes: list[str]) -> dict[str, bytes]:
        """Retrieve one or more files given a list of wanted hashes"""
