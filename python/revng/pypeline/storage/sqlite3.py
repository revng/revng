#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection, Mapping

from revng.pypeline.container import ConfigurationId
from revng.pypeline.model import ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.task import ObjectDependencies
from revng.pypeline.utils.registry import get_singleton

from .storage_provider import ContainerLocation, ProjectMetadata, SavePointsRange, StorageProvider
from .util import _OBJECTID_MAXSIZE, _REVNG_VERSION_PLACEHOLDER, check_kind_structure
from .util import check_object_id_supported_by_sql

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
);

DELETE FROM dependencies
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
        obj_id_ty: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        return [obj_id_ty.from_bytes(x[0]) for x in result]

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
        obj_id_ty: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        return {obj_id_ty.from_bytes(x[0]): x[1] for x in result}

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
        values: Mapping[ObjectID, bytes],
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
                        content,
                    ),
                )
            self._write_metadata(cursor)

    def invalidate(self, invalidation_list: ModelPathSet) -> None:
        if len(invalidation_list) == 0:
            return

        with self._cursor() as cursor:
            cursor.executescript(
                INVALIDATE_QUERY.format(
                    model_paths=",".join(f"'{path}'" for path in invalidation_list),
                    objectid_mask=_OBJECTID_MASK,
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
