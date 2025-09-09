#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from revng.pypeline.container import ContainerID
from revng.pypeline.model import ModelPath, ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies

from .storage_provider import ConfigurationId, ContainerLocation, ProjectMetadata, SavepointID
from .storage_provider import SavePointsRange, StorageProvider
from .util import _REVNG_VERSION_PLACEHOLDER


@dataclass
class DependencyEntry:
    savepoint_start: SavepointID
    savepoint_end: SavepointID
    container_id: ContainerID
    configuration_id: ConfigurationId
    object_id: ObjectID


class InMemoryStorageProvider(StorageProvider):
    """A simple in-memory storage provider for testing purposes.
    This is not thread-safe and should not be used in production.
    """

    def __init__(self):
        self.model = b""
        self.storage: dict[ContainerLocation, dict[ObjectID, bytes]] = {}
        self.dependencies: dict[str, list[DependencyEntry]] = defaultdict(list)
        self.last_change = datetime.now()

    def has(
        self,
        location: ContainerLocation,
        keys: Iterable[ObjectID],
    ) -> Iterable[ObjectID]:
        if location not in self.storage:
            return []
        storage = self.storage[location]
        return [key for key in keys if key in storage]

    def get(
        self,
        location: ContainerLocation,
        keys: Iterable[ObjectID],
    ) -> Mapping[ObjectID, bytes]:
        if location not in self.storage:
            raise KeyError(f"Savepoint {location} not found in storage.")
        storage = self.storage[location]
        return {k: storage[k] for k in keys if k in storage}

    def add_dependencies(
        self,
        savepoint_range: SavePointsRange,
        configuration_id: ConfigurationId,
        deps: ObjectDependencies,
    ) -> None:
        for container_name, obj, path in deps:
            self.dependencies[path].append(
                DependencyEntry(
                    savepoint_range.start,
                    savepoint_range.end,
                    container_name,
                    configuration_id,
                    obj,
                )
            )
        self.last_change = datetime.now()

    def put(
        self,
        location: ContainerLocation,
        values: Mapping[ObjectID, bytes],
    ) -> None:
        self.storage.setdefault(location, {})
        for key, value in values.items():
            self.storage[location][key] = value
        self.last_change = datetime.now()

    def _invalidate(self, path: ModelPath) -> None:
        for key, entries in self.dependencies.items():
            if key != path:
                continue

            # TODO: this double loop is very inefficient
            for entry in entries:
                for container_loc, objects in self.storage.items():
                    if (
                        container_loc.savepoint_id < entry.savepoint_start
                        or container_loc.savepoint_id > entry.savepoint_end
                        or container_loc.container_id != entry.container_id
                        or container_loc.configuration_id != entry.configuration_id
                    ):
                        continue
                    if entry.object_id in objects:
                        del objects[entry.object_id]

    def invalidate(self, invalidation_list: ModelPathSet) -> None:
        for path in invalidation_list:
            self._invalidate(path)
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
        self.storage.clear()
        self.dependencies.clear()
