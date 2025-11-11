#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from collections.abc import Buffer
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Iterable, Mapping

from revng import __version__ as revng_version
from revng.pypeline.container import ConfigurationId
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies

from .file_provider import FileRequest
from .storage_provider import ContainerLocation, FileStorageEntry, InvalidatedObjects, ProjectID
from .storage_provider import ProjectMetadata, SavePointsRange, StorageProvider
from .storage_provider import StorageProviderFactory
from .util import compute_hash


class NullStorageProviderFactory(StorageProviderFactory):
    def __init__(self, url: str):
        assert url == "null://"

    @classmethod
    def scheme(cls) -> str:
        return "null"

    @asynccontextmanager
    async def get(
        self,
        project_id: ProjectID | None,
        token: str | None,
        cache_dir: str | None,
    ) -> AsyncGenerator[StorageProvider]:
        yield NullStorageProvider()


class NullStorageProvider(StorageProvider):
    """The /dev/null of storage providers. It stores nothing and caches nothing.
    It just keeps track of the model, project ID, and last change time as those
    are required by the interface.
    """

    def __init__(self):
        self.model = b""
        self.last_change = datetime.now()

    def has(
        self,
        location: ContainerLocation,
        keys: Iterable[ObjectID],
    ) -> Iterable[ObjectID]:
        return []

    def get(
        self,
        location: ContainerLocation,
        keys: Iterable[ObjectID],
    ) -> Mapping[ObjectID, bytes]:
        assert not keys, "NullStorageProvider does not support get operation."
        return {}

    def add_dependencies(
        self,
        savepoint_range: SavePointsRange,
        configuration_id: ConfigurationId,
        deps: ObjectDependencies,
    ) -> None:
        self.last_change = datetime.now()

    def put(
        self,
        location: ContainerLocation,
        values: Mapping[ObjectID, Buffer],
    ) -> None:
        self.last_change = datetime.now()

    def invalidate(self, invalidation_list: ModelPathSet) -> InvalidatedObjects:
        self.last_change = datetime.now()
        return {}

    def get_epoch(self) -> int:
        return 0

    def get_model(self) -> tuple[bytes, int]:
        return (self.model, self.get_epoch())

    def set_model(self, new_model: bytes):
        self.model = new_model
        self.last_change = datetime.now()
        return 0

    def metadata(self) -> ProjectMetadata:
        """
        Fetch metadata about the current project
        """
        return ProjectMetadata(
            last_change=self.last_change,
            revng_version=revng_version,
        )

    def prune_objects(self):
        """
        Prunes all the objects (except metadata) from storage
        """

    def put_files_in_storage(self, files: list[FileStorageEntry]) -> list[str]:
        result = []
        for file in files:
            if file.path is not None:
                result.append(compute_hash(file.path))
            elif file.contents is not None:
                result.append(compute_hash(file.contents))
        return result

    def get_files_from_storage(self, requests: list[FileRequest]) -> dict[str, bytes]:
        raise ValueError("Unsupported")
