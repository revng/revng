#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import os
import shutil
import sqlite3
from collections import defaultdict
from collections.abc import Buffer
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection, Mapping

import yaml

from revng.pypeline.container import ConfigurationId
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies
from revng.pypeline.utils import cache_directory
from revng.pypeline.utils.registry import get_singleton

from .file_provider import FileRequest
from .storage_provider import ContainerLocation, FileStorageEntry, InvalidatedObjects
from .storage_provider import ProjectMetadata, SavePointsRange, StorageProvider
from .util import _OBJECTID_MAXSIZE, _REVNG_VERSION_PLACEHOLDER, check_kind_structure
from .util import check_object_id_supported_by_sql, compute_hash

# This is a binary mask that will be used for invalidation, thanks to the
# binary structure of ObjectID, all children are guaranteed to have the parent
# prefixed, so, to check for all children the check will be
# target_object_id <= checked_object_id <= CONCAT(target_object_id, _OBJECTID_MASK)  # noqa: E800
_OBJECTID_MASK = f"x'{"ff" * _OBJECTID_MAXSIZE}'"

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS project(
    id              TEXT PRIMARY KEY CHECK (id = 0),
    last_change     REAL,
    revng_version   TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS objects(
    savepoint_id         INT NOT NULL,
    container_id         TEXT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    object_id            BLOB NOT NULL,
    content              BLOB NOT NULL,
    PRIMARY KEY (savepoint_id, container_id, configuration_hash, object_id)
) STRICT;

CREATE INDEX IF NOT EXISTS savepoint_id_on_object ON objects(savepoint_id);
CREATE INDEX IF NOT EXISTS container_id_on_object ON objects(container_id);
CREATE INDEX IF NOT EXISTS configuration_hash_on_object ON objects(configuration_hash);
CREATE INDEX IF NOT EXISTS object_id_on_object ON objects(object_id);

CREATE TABLE IF NOT EXISTS dependencies(
    savepoint_id_start   INT NOT NULL,
    savepoint_id_end     INT NOT NULL,
    container_id         TEXT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    object_id            BLOB NOT NULL,
    model_path           TEXT NOT NULL,
    PRIMARY KEY (savepoint_id_start, savepoint_id_end, container_id,
                 configuration_hash, object_id, model_path)
) STRICT;

CREATE INDEX IF NOT EXISTS model_path_on_dependencies ON dependencies(model_path);
"""

HAS_QUERY = """
SELECT object_id FROM objects
WHERE
  savepoint_id = ?
  AND container_id = ?
  AND configuration_hash = ?
  AND object_id IN ({id_list})"""

GET_QUERY = """
SELECT object_id, content FROM objects
WHERE
  savepoint_id = ?
  AND container_id = ?
  AND configuration_hash = ?
  AND object_id IN ({id_list})"""

PUT_QUERY = """
REPLACE INTO objects(savepoint_id, container_id, configuration_hash, object_id, content)
VALUES (?, ?, ?, ?, ?)
"""

PUT_DEPENDENCIES_QUERY = "REPLACE INTO dependencies VALUES (?, ?, ?, ?, ?, ?)"

# Invalidation query, this retrieves the paths stored in `dependencies` that match.
# Given the records found, the following rules must all be met when selecting
# records from objects to be deleted:
# 1. Have the savepoint be in the savepoint range of the dependency
# 2. Have the configuration_hash match
# 3. Have the object_id be related to the dependency, this means that, either:
#    1. The objects.object_id is the same or a child of dependencies.object_id
#       This is simplified by the structure of ObjectID, so this becomes a
#       range comparison
#       `dependencies.object_id <= objects.object_id <= (dependencies.object_id + mask)`
#    2. The object.object_id is a parent of dependencies.object_id. In the
#       general case this would require generating the query dynamically. For
#       the time being this is specialized for the current structure where
#       there's only children of root, so the check becomes:
#       `dependencies.object_id != x'' AND objects.object_id = x''`
INVALIDATE_QUERY = """
DELETE FROM objects
WHERE rowid IN (
    SELECT objects.rowid
    FROM objects
    JOIN dependencies
    WHERE dependencies.model_path IN ({model_paths})
          AND (
            (
              objects.object_id >= dependencies.object_id
              AND objects.object_id <= CAST((dependencies.object_id || {objectid_mask}) AS BLOB)
            ) OR (
              dependencies.object_id != x'' AND objects.object_id = x''
            )
          )
          AND objects.configuration_hash = dependencies.configuration_hash
          AND objects.savepoint_id >= dependencies.savepoint_id_start
          AND objects.savepoint_id <= dependencies.savepoint_id_end
)
RETURNING object_id, container_id, savepoint_id, configuration_hash;
"""

DELETE_MODEL_PATHS = """DELETE FROM dependencies
WHERE dependencies.model_path IN ({model_paths});
"""


class CursorWrapper:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.cursor: sqlite3.Cursor | None = None

    def __enter__(self) -> sqlite3.Cursor:
        assert self.cursor is None
        self.cursor = self.connection.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.cursor is not None
        if exc_type is not None:
            self.connection.rollback()
        else:
            self.connection.commit()
        self.cursor.close()
        return False


class SQlite3StorageProvider(StorageProvider):
    """StorageProvider implementation with backing sqlite3 db"""

    def __init__(self, db_path: str, model_path: str | Path):
        check_kind_structure()
        self._model_path = Path(model_path)
        self._model_directory = self._model_path.parent.resolve()
        self._connection = sqlite3.connect(db_path, autocommit=False)
        self._connection.commit()
        self._init_tables()

    def _cursor(self) -> CursorWrapper:
        return CursorWrapper(self._connection)

    def _init_tables(self):
        with self._cursor() as cursor:
            cursor.executescript(CREATE_TABLES)

    def _write_metadata(self, cursor: sqlite3.Cursor):
        cursor.execute(
            "REPLACE INTO project VALUES (0, ?, ?)",
            (datetime.now().timestamp(), _REVNG_VERSION_PLACEHOLDER),
        )

    def has(
        self,
        location: ContainerLocation,
        keys: Collection[ObjectID],
    ) -> list[ObjectID]:
        if len(keys) == 0:
            return []

        # NOTE: possible SQL injection, sqlite has a limit on parameters. If it
        # can't be avoided chunk selects by 999 values
        id_list = ",".join([f"x'{key.to_bytes().hex()}'" for key in keys])
        with self._cursor() as cursor:
            cursor.execute(
                HAS_QUERY.format(id_list=id_list),
                (location.savepoint_id, location.container_id, location.configuration_id),
            )
            result = cursor.fetchall()

        obj_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        return [obj_id_type.from_bytes(x[0]) for x in result]

    def get(
        self,
        location: ContainerLocation,
        keys: Collection[ObjectID],
    ) -> dict[ObjectID, bytes]:
        if len(keys) == 0:
            return {}

        # NOTE: possible SQL injection, sqlite has a limit on parameters. If it
        # can't be avoided chunk selects by 999 values
        id_list = ",".join([f"x'{key.to_bytes().hex()}'" for key in keys])
        with self._cursor() as cursor:
            cursor.execute(
                GET_QUERY.format(id_list=id_list),
                (location.savepoint_id, location.container_id, location.configuration_id),
            )
            result = cursor.fetchall()
        obj_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        return {obj_id_type.from_bytes(x[0]): x[1] for x in result}

    def add_dependencies(
        self,
        savepoint_range: SavePointsRange,
        configuration_id: ConfigurationId,
        deps: ObjectDependencies,
    ) -> None:
        with self._cursor() as cursor:
            for container_id, object_id, model_path in deps:
                check_object_id_supported_by_sql(object_id)
                cursor.execute(
                    PUT_DEPENDENCIES_QUERY,
                    (
                        savepoint_range.start,
                        savepoint_range.end,
                        container_id,
                        configuration_id,
                        object_id.to_bytes(),
                        model_path,
                    ),
                )
            self._write_metadata(cursor)

    def put(
        self,
        location: ContainerLocation,
        values: Mapping[ObjectID, Buffer],
    ) -> None:
        with self._cursor() as cursor:
            for object_id, content in values.items():
                cursor.execute(
                    PUT_QUERY,
                    (
                        location.savepoint_id,
                        location.container_id,
                        location.configuration_id,
                        object_id.to_bytes(),
                        bytes(content),
                    ),
                )
            self._write_metadata(cursor)

    def invalidate(self, invalidation_list: ModelPathSet) -> InvalidatedObjects:
        if len(invalidation_list) == 0:
            return {}

        object_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        invalidated: InvalidatedObjects = defaultdict(set)
        joined_paths = ",".join(f"'{path}'" for path in invalidation_list)
        with self._cursor() as cursor:
            cursor.execute(
                INVALIDATE_QUERY.format(model_paths=joined_paths, objectid_mask=_OBJECTID_MASK)
            )
            for row in cursor:
                object_id = row[0]
                location = ContainerLocation(
                    container_id=row[1],
                    savepoint_id=row[2],
                    configuration_id=row[3],
                )
                invalidated[location].add(object_id_type.from_bytes(object_id))

            cursor.execute(DELETE_MODEL_PATHS.format(model_paths=joined_paths))
            self._write_metadata(cursor)

        return dict(invalidated)

    def prune_objects(self):
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM objects")
            cursor.execute("DELETE FROM dependencies")
            self._write_metadata(cursor)

    def get_model(self) -> bytes:
        return self._model_path.read_bytes()

    def set_model(self, new_model: bytes):
        self._model_path.write_bytes(new_model)
        with self._cursor() as cursor:
            self._write_metadata(cursor)

    def metadata(self) -> ProjectMetadata:
        with self._cursor() as cursor:
            cursor.execute("SELECT last_change, revng_version FROM project WHERE id is NULL")
            result = cursor.fetchone()

        return ProjectMetadata(
            last_change=datetime.fromtimestamp(result[0], timezone.utc),
            revng_version=result[1],
        )

    def put_files_in_storage(self, files: list[FileStorageEntry]) -> list[str]:
        result = []
        for file in files:
            if file.contents is not None:
                file_path = self._model_directory / file.name
                file_path.write_bytes(file.contents)
                hash_ = compute_hash(file.contents)

            elif file.path is not None:
                if file_path.parent.resolve() != self._model_directory:
                    # If here, the file is not in the directory where the model
                    # is present, copy it there
                    file_path = self._model_directory / file.path.name
                    shutil.copy2(file.path, file_path)
                else:
                    file_path = file.path

                hash_ = compute_hash(file_path)

            self._write_link_file(file_path, hash_)
            result.append(hash_)

        return result

    def get_files_from_storage(self, requests: list[FileRequest]) -> dict[str, bytes]:
        return {r.hash: self._find_file(r).read_bytes() for r in requests}

    def _write_link_file(self, path: Path, hash_: str):
        link_file = cache_directory() / f"resources/{hash_}.link"
        link_file.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        if link_file.is_file():
            data = yaml.safe_load(link_file.read_text())

        paths = {Path(p) for p in data.get("PathHints", [])}
        paths.add(path.resolve())
        paths.add(path.relative_to(self._model_directory, walk_up=True))

        mtimes = []
        for path_element in paths:
            mtimes.append(path_element.stat().st_mtime)

        data = {"PathHints": [str(p) for p in paths], "ModifiedTime": max(mtimes)}
        link_file.write_text(yaml.safe_dump(data))

    def _find_file(self, request: FileRequest) -> Path:
        link_file = cache_directory() / f"resources/{request.hash}.link"
        if link_file.is_file():
            data = yaml.safe_load(link_file.read_text())

            # Read the `PathHints` list, normalize the entries to all absolute paths
            found_paths: list[Path] = []
            for path in data["PathHints"]:
                path_path = Path(path)
                if not path_path.is_absolute():
                    path_path = (self._model_directory / path_path).resolve()

                if path_path.is_file():
                    found_paths.append(path_path)

            # Try and find a file from found_paths that matches the `ModifiedTime`
            mtime = data["ModifiedTime"]
            for path in found_paths:
                if path.stat().st_mtime == mtime:
                    return path

            # If here none of the found_paths matched mtime, try and update it
            for path in found_paths:
                if compute_hash(path) == request.hash:
                    data["ModifiedTime"] = path.stat().st_mtime
                    link_file.write_text(yaml.safe_dump(data))
                    return path

        # If here, none of the paths in found_paths had a matching hash
        # First check if there is a file in the model directory that matches
        if request.name is not None:
            maybe_file = self._model_directory / request.name
            if self._compare_file(maybe_file, request):
                self._write_link_file(maybe_file, request.hash)
                return maybe_file

        # If here, as a last resort, scan the entire model directory to try and
        # find the file
        with os.scandir(self._model_directory) as scan_iter:
            for entry in scan_iter:
                entry_path = Path(entry.path)
                if entry.is_file() and self._compare_file(entry_path, request):
                    self._write_link_file(entry_path, request.hash)
                    return entry_path

        # If here, no file has been found, throw an exception
        raise ValueError("Could not find a suitable file")

    @staticmethod
    def _compare_file(path: Path, request: FileRequest) -> bool:
        return (
            path.is_file()
            and (request.size is None or path.stat().st_size == request.size)
            and compute_hash(path) == request.hash
        )
