#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from collections.abc import Buffer
from datetime import datetime
from typing import Iterable, Mapping

from revng.pypeline.container import ConfigurationId
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies

from .file_provider import FileRequest
from .storage_provider import ContainerLocation, FileStorageEntry, InvalidatedObjects
from .storage_provider import ProjectMetadata, SavePointsRange, StorageProvider
from .util import _REVNG_VERSION_PLACEHOLDER, compute_hash


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

    def get_model(self) -> bytes:
        return self.model

    def set_model(self, new_model: bytes):
        self.model = new_model
        self.last_change = datetime.now()

    def metadata(self) -> ProjectMetadata:
        """
        Fetch metadata about the current project
        """
        return ProjectMetadata(
            last_change=self.last_change,
            revng_version=_REVNG_VERSION_PLACEHOLDER,
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
