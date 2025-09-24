#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping

from revng.pypeline.container import ConfigurationId
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies

from .storage_provider import ContainerLocation, ProjectID, ProjectMetadata, SavePointsRange
from .storage_provider import StorageProvider, StorageProviderFactory
from .util import _REVNG_VERSION_PLACEHOLDER


class NullStorageProviderFactory(StorageProviderFactory):
    def __init__(self, url: str):
        assert url == ""

    @classmethod
    def protocol(cls) -> str:
        return "null"

    def get(
        self,
        project_id: ProjectID | None,
        token: str | None,
        cache_dir: str | None,
    ) -> StorageProvider:
        return NullStorageProvider()


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
        values: Mapping[ObjectID, bytes],
    ) -> None:
        self.last_change = datetime.now()

    def invalidate(self, invalidation_list: ModelPathSet) -> None:
        self.last_change = datetime.now()

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
