#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from collections.abc import Buffer
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Iterable, Mapping

from revng.pypeline import __version__ as version
from revng.pypeline.model import Model, ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.task.pipe import PipeCustomInvalidation
from revng.pypeline.utils.registry import get_singleton

from .file_provider import FileRequest
from .storage_provider import ContainerLocation, FileStorageEntry, LockType, ObjectsToInvalidate
from .storage_provider import PipeDependencies, ProjectID, ProjectMetadata, SetModelResult
from .storage_provider import StorageProvider, StorageProviderFactory
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
        base_directory: Path,
        pipeline: Pipeline,
        lock_type: LockType,
        project_id: ProjectID | None,
        token: str | None,
        cache_dir: str | None,
    ) -> AsyncGenerator[StorageProvider]:
        yield NullStorageProvider()

    def get_notification_websocket(self) -> str | None:
        return None


class NullStorageProvider(StorageProvider):
    """The /dev/null of storage providers. It stores nothing and caches nothing.
    It just keeps track of the model, project ID, and last change time as those
    are required by the interface.
    """

    def __init__(self):
        self.model = get_singleton(Model)()
        self.last_fetch = datetime.fromtimestamp(0, timezone.utc)
        self.last_object_save = datetime.fromtimestamp(0, timezone.utc)
        self.last_model_save = datetime.fromtimestamp(0, timezone.utc)

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

    def add_objects(
        self,
        dependencies: list[PipeDependencies],
        objects: Mapping[ContainerLocation, Mapping[ObjectID, Buffer]],
    ) -> None:
        self.last_object_save = datetime.now(timezone.utc)

    def get_epoch(self) -> int:
        return 0

    def get_model(self) -> tuple[Model, int]:
        return (self.model.clone(), self.get_epoch())

    def set_model(
        self,
        new_model: Model,
        changed_paths: ModelPathSet,
        custom_invalidations: list[ObjectsToInvalidate],
        model_bytes: bytes | None = None,
    ) -> SetModelResult:
        self.model = new_model.clone()
        self.last_model_save = datetime.now(timezone.utc)
        return SetModelResult(0, {})

    def metadata(self) -> ProjectMetadata:
        """
        Fetch metadata about the current project
        """
        return ProjectMetadata(
            version=version,
            pipeline_description_hash="",
            last_fetch=self.last_fetch,
            last_object_save=self.last_object_save,
            last_model_save=self.last_model_save,
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

    def get_custom_invalidation_data(
        self, pipe_id: int, configuration_hash: str
    ) -> PipeCustomInvalidation:
        raise ValueError("Unsupported")
