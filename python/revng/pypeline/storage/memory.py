#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from revng.pypeline.container import ContainerID
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies

from .storage_provider import ConfigurationId, ContainerLocation, ProjectMetadata, SavepointID
from .storage_provider import SavePointsRange, StorageProvider
from .util import _REVNG_VERSION_PLACEHOLDER, check_kind_structure


@dataclass(frozen=True)
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
        check_kind_structure()
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

    def invalidate(self, invalidation_list: ModelPathSet):
        invalidated: dict[ContainerLocation, set[ObjectID]] = defaultdict(set)

        # Set of entries that will be collected from self.dependencies
        object_to_delete: set[DependencyEntry] = set()
        # Keys that exist on self.dependencies that will be deleted
        paths_to_delete: list[str] = []

        # Retrieve entries from self.dependencies that match the provided paths
        for path in invalidation_list:
            if path in self.dependencies:
                paths_to_delete.append(path)
                object_to_delete.update(self.dependencies[path])

        # For each DependencyEntry, find the matching entries in self.storage
        # TODO: this double loop is very inefficient
        for entry in object_to_delete:
            for container_loc, objects in self.storage.items():
                # For an entry to match, the savepoint must be in the savepoint
                # range and the configuration_id must match
                if (
                    container_loc.savepoint_id < entry.savepoint_start
                    or container_loc.savepoint_id > entry.savepoint_end
                    or container_loc.configuration_id != entry.configuration_id
                ):
                    continue

                # Check that the object is related (the same object, a parent
                # or a child) to the invalidated object
                for object_ in objects:
                    if entry.object_id.is_related(object_):
                        invalidated[container_loc].add(object_)

        # Actually delete the data
        for container_loc, objects_to_delete in invalidated.items():
            self.storage[container_loc] = {
                k: v for k, v in self.storage[container_loc].items() if k not in objects_to_delete
            }

        # Delete the paths that were previously found
        for path in paths_to_delete:
            del self.dependencies[path]

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
