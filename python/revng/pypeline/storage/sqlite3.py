#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from collections.abc import Buffer
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import yaml

from revng.pypeline.container import ConfigurationId
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies
from revng.pypeline.utils import cache_directory
from revng.pypeline.utils.registry import get_singleton

from .file_storage import FileRequest
from .storage_provider import ContainerLocation, FileStorageEntry, ProjectMetadata
from .storage_provider import SavePointsRange, StorageProvider
from .util import _REVNG_VERSION_PLACEHOLDER

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
    object_id            TEXT NOT NULL,
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
    object_id            TEXT NOT NULL,
    model_path_hash      TEXT NOT NULL,
    PRIMARY KEY (savepoint_id_start, savepoint_id_end, container_id,
                 configuration_hash, object_id, model_path_hash)
) STRICT;

CREATE INDEX IF NOT EXISTS model_path_hash_on_dependencies ON dependencies(model_path_hash);
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

INVALIDATE_QUERY = """
DELETE FROM objects
WHERE rowid IN (
    SELECT objects.rowid
    FROM objects
    JOIN dependencies
    WHERE dependencies.model_path_hash IN ({model_paths})
          AND objects.container_id = dependencies.container_id
          AND objects.configuration_hash = dependencies.configuration_hash
          AND objects.savepoint_id >= dependencies.savepoint_id_start
          AND objects.savepoint_id <= dependencies.savepoint_id_end
);

DELETE FROM dependencies
WHERE dependencies.model_path_hash IN ({model_paths});
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
        keys: Iterable[ObjectID],
    ) -> Iterable[ObjectID]:
        # NOTE: possible SQL injection, sqlite has a limit on parameters. If it
        # can't be avoided chunk selects by 999 values
        id_list = ",".join([f"'{key!s}'" for key in keys])
        with self._cursor() as cursor:
            cursor.execute(
                HAS_QUERY.format(id_list=id_list),
                (location.savepoint_id, location.container_id, location.configuration_id),
            )
            result = cursor.fetchall()
        obj_id_ty: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        return [obj_id_ty.deserialize(x[0]) for x in result]

    def get(
        self,
        location: ContainerLocation,
        keys: Iterable[ObjectID],
    ) -> Mapping[ObjectID, bytes]:
        # NOTE: possible SQL injection, sqlite has a limit on parameters. If it
        # can't be avoided chunk selects by 999 values
        id_list = ",".join([f"'{key!s}'" for key in keys])
        with self._cursor() as cursor:
            cursor.execute(
                GET_QUERY.format(id_list=id_list),
                (location.savepoint_id, location.container_id, location.configuration_id),
            )
            result = cursor.fetchall()
        obj_id_ty: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        return {obj_id_ty.deserialize(x[0]): x[1] for x in result}

    def add_dependencies(
        self,
        savepoint_range: SavePointsRange,
        configuration_id: ConfigurationId,
        deps: ObjectDependencies,
    ) -> None:
        with self._cursor() as cursor:
            for container_id, object_id, model_path in deps:
                cursor.execute(
                    "REPLACE INTO dependencies VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        savepoint_range.start,
                        savepoint_range.end,
                        container_id,
                        configuration_id,
                        object_id.serialize(),
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
                        object_id.serialize(),
                        bytes(content),
                    ),
                )
            self._write_metadata(cursor)

    def invalidate(self, invalidation_list: ModelPathSet) -> None:
        with self._cursor() as cursor:
            cursor.executescript(
                INVALIDATE_QUERY.format(
                    model_paths=",".join(f"'{path}'" for path in invalidation_list)
                )
            )
            self._write_metadata(cursor)

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

    def put_files_in_file_storage(self, files: list[FileStorageEntry]) -> list[str]:
        result = []
        for file in files:
            if file.contents is not None:
                file_path = self._model_directory / file.name
                file_path.write_bytes(file.contents)
                hash_ = hashlib.sha256(file.contents).hexdigest()

            elif file.path is not None:
                if file_path.parent.resolve() != self._model_directory:
                    # If here, the file is not in the directory where the model
                    # is present, copy it there
                    file_path = self._model_directory / file.path.name
                    shutil.copy2(file.path, file_path)
                else:
                    file_path = file.path

                hash_ = self._get_file_hash(file_path)

            self._write_link_file(file_path, hash_)
            result.append(hash_)

        return result

    def get_files_from_file_storage(self, requests: list[FileRequest]) -> dict[str, bytes]:
        return {r.hash: self._find_file(r).read_bytes() for r in requests}

    def _write_link_file(self, path: Path, hash_: str):
        link_file = cache_directory() / f"resources/{hash_}.link"
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
                if self._get_file_hash(path) == request.hash:
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

    @classmethod
    def _compare_file(cls, path: Path, request: FileRequest) -> bool:
        return (
            path.is_file()
            and (request.size is None or path.stat().st_size == request.size)
            and cls._get_file_hash(path) == request.hash
        )

    @staticmethod
    def _get_file_hash(path: Path):
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
