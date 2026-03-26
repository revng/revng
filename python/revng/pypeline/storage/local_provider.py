#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import os
import shutil
import sqlite3
from collections import defaultdict
from collections.abc import Buffer
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import AsyncGenerator, Collection, Mapping
from uuid import uuid4

import yaml

from revng.pypeline import __version__ as version
from revng.pypeline.container import ConfigurationId
from revng.pypeline.model import Model, ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.task.pipe import ObjectDependencies, PipeCustomInvalidation
from revng.pypeline.utils import Locked, crypto_hash
from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.registry import get_singleton

from .file_provider import FileRequest
from .storage_provider import ContainerLocation, FileStorageEntry, InvalidatedObjects
from .storage_provider import ObjectsToInvalidate, ProjectID, ProjectMetadata, SavePointsRange
from .storage_provider import StorageProvider, StorageProviderFactory
from .util import _OBJECTID_MAXSIZE, check_kind_structure, check_object_id_supported_by_sql
from .util import compute_hash

# This is a binary mask that will be used for invalidation, thanks to the
# binary structure of ObjectID, all children are guaranteed to have the parent
# prefixed, so, to check for all children the check will be
# target_object_id <= checked_object_id <= CONCAT(target_object_id, _OBJECTID_MASK)  # noqa: E800
_OBJECTID_MASK = f"x'{"ff" * _OBJECTID_MAXSIZE}'"

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS project(
    id              TEXT PRIMARY KEY CHECK (id = 0),
    last_change     REAL,
    epoch           INT NOT NULL,
    version         TEXT
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

CREATE TABLE IF NOT EXISTS custom_dependencies(
    pipe_id              INT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    argument_index       INT NOT NULL,
    object_id            BLOB NOT NULL,
    data                 BLOB NOT NULL,
    PRIMARY KEY (pipe_id, configuration_hash)
) STRICT;

CREATE INDEX IF NOT EXISTS custom_dependencies_index
    ON custom_dependencies(pipe_id, configuration_hash);
"""

CREATE_TABLE_INVALIDATE = """
CREATE TEMPORARY TABLE model_paths_{uuid}(
    path   TEXT NOT NULL
) STRICT;

CREATE TEMPORARY TABLE additional_objects_{uuid}(
    savepoint_id_start   INT NOT NULL,
    savepoint_id_end     INT NOT NULL,
    container_id         TEXT NOT NULL,
    configuration_hash   TEXT NOT NULL,
    object_id            BLOB NOT NULL,
    PRIMARY KEY (savepoint_id_start, savepoint_id_end, container_id,
                 configuration_hash, object_id)
) STRICT;

CREATE TEMPORARY TABLE invalidated_objects_{uuid}(
    object_row_id   INT NOT NULL
) STRICT;
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

# Invalidation queries, these takes care, given populated tables `model_paths`
# and `additional_objects`, to remove all the applicable entries from the DB.
# The structure is such that only the invalidation data is returned to python,
# everything else lives exclusively on the DB; this saves up in round-trips.
# Invalidation happens with 3 queries:
# 1. Match the entries from the `objects` table that need to be removed (see
#    below), store their `rowid` in the `invalidated_objects` temporary table
# 2. Delete from the `dependencies` table all the entries that have been
#    generated by objects from (1). Since we already recorded which objects we
#    want to delete the entries from this table are no longer needed.
# 3. Actually delete the entries from the `objects` table and return the list
#    to python. This is in a separate variable due to technical reasons.
#
# The logic that decides, given a set of `model_paths` and
# `additional_objects`, which entries from `objects` will be delete is as
# follows:
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
INSERT INTO invalidated_objects_{uuid}
    SELECT DISTINCT objects.rowid
    FROM objects
    JOIN (
            SELECT savepoint_id_start, savepoint_id_end, configuration_hash, object_id
            FROM dependencies
            WHERE dependencies.model_path IN (SELECT path FROM model_paths_{uuid})
        UNION
            SELECT savepoint_id_start, savepoint_id_end, configuration_hash, object_id
            FROM additional_objects_{uuid}
    ) AS dependencies
    WHERE (
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
;

DELETE FROM dependencies
WHERE rowid in (
    SELECT DISTINCT dependencies.rowid
    FROM dependencies
    JOIN (
        SELECT savepoint_id, container_id, configuration_hash, object_id
        FROM objects
        WHERE rowid IN (SELECT object_row_id FROM invalidated_objects_{uuid})
    ) AS invalidated_objects
    WHERE
        dependencies.savepoint_id_start = invalidated_objects.savepoint_id
        AND dependencies.container_id = invalidated_objects.container_id
        AND dependencies.configuration_hash = invalidated_objects.configuration_hash
        AND dependencies.object_id = invalidated_objects.object_id
);
"""

# Second part of `INVALIDATE_QUERY`, this is a separate variable due to a
# limitation of the `sqlite3` module that does not return result rows when
# `executescript` is used
INVALIDATE_QUERY_2 = """
DELETE FROM objects
WHERE rowid IN (SELECT object_row_id FROM invalidated_objects_{uuid})
RETURNING object_id, container_id, savepoint_id, configuration_hash;
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


class LocalStorageProviderFactory(StorageProviderFactory):
    def __init__(self, url: str):
        assert url == "local://" or "local://?inline"
        # TODO: use urlparse if more options are introduced
        self.inline = url == "local://?inline"
        self.providers: Locked[dict[ProjectID | None, Locked[LocalStorageProvider]]] = Locked({})

    @classmethod
    def scheme(cls) -> str:
        return "local"

    def _create_provider(
        self,
        base_directory: Path,
        project_id: ProjectID | None,
        token: str | None,
        cache_dir: str,
    ) -> LocalStorageProvider:
        # Figure out how the model should be name
        model_type = get_singleton(Model)  # type: ignore [type-abstract]
        model_name = model_type.model_name()

        # Find the model in the current directory or any of its parents

        directory = base_directory
        while True:
            pypeline_logger.debug_log(f'Searching for model at "{directory / model_name}"')
            if (directory / model_name).exists():
                break
            if directory == directory.parent:
                raise FileNotFoundError(f'Model "{model_name}" not found')
            directory = directory.parent

        model_path = directory / model_name
        pypeline_logger.debug_log(f'Model "{model_name}" found at "{model_path}"')
        # Compute the hash of the model path as a tentative unique identifier for the project
        # TODO: we are relying on the *absolute* model path, which means that if the
        # user moves the project around, it will be treated as a different project
        # and caches will be recomputed, and most importantly, if someone deletes
        # the project and creates a new one at the same path, it will reuse the
        # old cache.
        if self.inline:
            cache_path = (directory / ".cache").resolve()
            cache_path.mkdir(parents=True, exist_ok=True)
            db_path = cache_path / "data.sqlite"
        else:
            cache_path = Path(cache_dir)
            db_name = crypto_hash(str(model_path)) + ".sqlite"
            db_path = Path(cache_dir) / db_name

        pypeline_logger.debug_log(f'Using DB "{db_path}"')
        return LocalStorageProvider(db_path, model_path, cache_path)

    @asynccontextmanager
    async def get(
        self,
        base_directory: Path,
        project_id: ProjectID | None,
        token: str | None,
        cache_dir: str | None,
    ) -> AsyncGenerator[StorageProvider]:
        assert cache_dir is not None, "Cache directory must be provided"

        # Get or create the provider for the given project ID
        async with self.providers() as providers:
            project_provider: Locked[LocalStorageProvider] | None = providers.get(project_id)
            # If the provider is not found, create a new one and put it in a lock
            if project_provider is None:
                project_provider = Locked(
                    self._create_provider(
                        base_directory=base_directory,
                        project_id=project_id,
                        token=token,
                        cache_dir=cache_dir,
                    )
                )
                providers[project_id] = project_provider

        # Release the global lock and acquire the project-specific one so other
        # projects can proceed in parallel
        async with project_provider() as provider:
            yield provider


TemporaryProviderTuple = tuple["LocalStorageProvider", TemporaryDirectory]


class TemporaryLocalStorageProviderFactory(StorageProviderFactory):
    def __init__(self, url: str):
        assert url == "temporary://"
        self.providers: Locked[dict[ProjectID | None, Locked[TemporaryProviderTuple]]] = Locked({})

    @classmethod
    def scheme(cls) -> str:
        return "temporary"

    @asynccontextmanager
    async def get(
        self,
        base_directory: Path,
        project_id: ProjectID | None,
        token: str | None,
        cache_dir: str | None,
    ) -> AsyncGenerator[StorageProvider]:
        model_type = get_singleton(Model)  # type: ignore [type-abstract]
        model_name = model_type.model_name()

        # Get or create the provider for the given project ID
        async with self.providers() as providers:
            project_provider: Locked[TemporaryProviderTuple] | None = providers.get(project_id)
            # If the provider is not found, create a new one and put it in a lock
            if project_provider is None:
                temporary_dir = TemporaryDirectory()
                temp_dir_path = Path(temporary_dir.name)
                (temp_dir_path / model_name).touch()
                (temp_dir_path / "cache").mkdir()

                storage_provider = LocalStorageProvider(
                    db_path=temp_dir_path / "db.sqlite",
                    model_path=temp_dir_path / model_name,
                    cache_dir=temp_dir_path / "cache",
                )
                project_provider = Locked((storage_provider, temporary_dir))
                providers[project_id] = project_provider

        # Release the global lock and acquire the project-specific one so other
        # projects can proceed in parallel
        async with project_provider() as provider:
            yield provider[0]


class LocalStorageProvider(StorageProvider):
    """StorageProvider implementation with backing sqlite3 db"""

    def __init__(self, db_path: str | Path, model_path: Path, cache_dir: Path):
        check_kind_structure()
        self._model_path = model_path
        self._model_directory = self._model_path.parent.resolve()
        self._connection = sqlite3.connect(db_path, autocommit=False)
        self._connection.commit()
        self._init_tables()
        self.epoch = self._get_epoch()
        self._cache_dir = cache_dir
        self._model_type = get_singleton(Model)  # type: ignore[type-abstract]

    def _cursor(self) -> CursorWrapper:
        return CursorWrapper(self._connection)

    def _init_tables(self):
        with self._cursor() as cursor:
            cursor.executescript(CREATE_TABLES)

    def _write_metadata(self, cursor: sqlite3.Cursor):
        cursor.execute(
            "REPLACE INTO project VALUES (0, ?, ?, ?)",
            (datetime.now().timestamp(), self.epoch, version),
        )

    def _get_epoch(self) -> int:
        # Try to get the epoch from the DB
        with self._cursor() as cursor:
            cursor.execute("SELECT epoch FROM project WHERE id is 0")
            result = cursor.fetchone()
            if result is not None:
                return result[0]
            # Otherwise we have to write the metadata at least once
            self.epoch = 0
            self._write_metadata(cursor)
            return self.epoch

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

    def invalidate(
        self, invalidation_list: ModelPathSet, additional_objects: list[ObjectsToInvalidate]
    ) -> InvalidatedObjects:
        if len(invalidation_list) == 0 and len(additional_objects) == 0:
            return {}

        object_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        # Generate a random UUID, this will be needed to create a unique temporary table
        table_uuid = uuid4().hex
        # The result that will be returned by this function
        invalidated: InvalidatedObjects = defaultdict(set)
        with self._cursor() as cursor:
            # Create the required tables for invalidations, all prefixed with
            # UUID, these are:
            # * model_paths_{UUID}: input `model_paths`
            # * additional_objects_{UUID}: input `additional_objects`
            # * invalidated_objects_{UUID}: the list of rowids that have been
            #     invalidated. This is used internally by the queries.
            cursor.executescript(CREATE_TABLE_INVALIDATE.format(uuid=table_uuid))
            # Pour our input data into the temporary tables
            for path in invalidation_list:
                cursor.execute(f"REPLACE INTO model_paths_{table_uuid} VALUES (?)", (path,))
            for object_set in additional_objects:
                for object_ in object_set.objects:
                    cursor.execute(
                        f"REPLACE INTO additional_objects_{table_uuid} VALUES (?, ?, ?, ?, ?)",
                        (
                            object_set.savepoint_range.start,
                            object_set.savepoint_range.end,
                            object_set.container_id,
                            object_set.configuration_id,
                            object_.to_bytes(),
                        ),
                    )

            # Run the actual invalidation (see comment above the
            # `INVALIDATE_QUERY` for an explanation of how it selects the
            # objects to remove)
            cursor.executescript(
                INVALIDATE_QUERY.format(objectid_mask=_OBJECTID_MASK, uuid=table_uuid)
            )
            # Due to a limitation of sqlite3 only `execute` returns rows,
            # whereas executescript does not
            cursor.execute(INVALIDATE_QUERY_2.format(uuid=table_uuid))
            # Read the returned rows (the invalidated objects) and insert them
            # into the `invalidated` dictionary
            for row in cursor:
                object_id = row[0]
                location = ContainerLocation(
                    container_id=row[1],
                    savepoint_id=row[2],
                    configuration_id=row[3],
                )
                invalidated[location].add(object_id_type.from_bytes(object_id))

            # DROP the temporary tables as they are no longer needed
            cursor.execute(f"DROP TABLE additional_objects_{table_uuid};")
            cursor.execute(f"DROP TABLE model_paths_{table_uuid};")
            cursor.execute(f"DROP TABLE invalidated_objects_{table_uuid};")
            # Write the last_change field
            self._write_metadata(cursor)
        return dict(invalidated)

    def prune_objects(self):
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM objects")
            cursor.execute("DELETE FROM dependencies")
            cursor.execute("DELETE FROM custom_dependencies")
            self._write_metadata(cursor)

    def get_epoch(self) -> int:
        return self.epoch

    def get_model(self) -> tuple[Model, int]:
        model, changed = self._model_type.deserialize(self._model_path.read_bytes())
        if changed:
            self.prune_objects()
            self._write_model(model)
        return (model, self.epoch)

    def set_model(self, new_model: Model) -> int:
        # Check if the model was modified
        current_model, _ = self._model_type.deserialize(self._model_path.read_bytes())
        if current_model == new_model:
            return self.epoch
        # if so, write the new model and update the epoch
        return self._write_model(new_model)

    def _write_model(self, new_model: Model) -> int:
        self._model_path.write_bytes(new_model.serialize())
        with self._cursor() as cursor:
            self.epoch += 1
            self._write_metadata(cursor)
            return self.epoch

    def metadata(self) -> ProjectMetadata:
        with self._cursor() as cursor:
            cursor.execute("SELECT last_change, version FROM project WHERE id is 0")
            result = cursor.fetchone()

        return ProjectMetadata(
            last_change=datetime.fromtimestamp(result[0], timezone.utc),
            version=result[1],
        )

    def put_files_in_storage(self, files: list[FileStorageEntry]) -> list[str]:
        result = []
        for file in files:
            if file.contents is not None:
                file_path = self._model_directory / file.name
                file_path.write_bytes(file.contents)
                hash_ = compute_hash(file.contents)

            elif file.path is not None:
                if file.path.parent.resolve() != self._model_directory:
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
        link_file = self._cache_dir / f"resources/{hash_}.link"
        link_file.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        if link_file.is_file():
            data = yaml.safe_load(link_file.read_text())

        paths = {Path(p) for p in data.get("PathHints", [])}
        paths.add(path.resolve())
        paths.add(path.relative_to(self._model_directory, walk_up=True))

        mtimes = [data.get("ModifiedTime", 0)]
        for path_element in paths:
            try:
                stat = path_element.stat()
            except (OSError, ValueError):
                continue

            mtimes.append(stat.st_mtime)

        data = {"PathHints": [str(p) for p in paths], "ModifiedTime": max(mtimes)}
        link_file.write_text(yaml.safe_dump(data))

    def _find_file(self, request: FileRequest) -> Path:
        link_file = self._cache_dir / f"resources/{request.hash}.link"
        if link_file.is_file():
            data = yaml.safe_load(link_file.read_text())

            # Read the `PathHints` list, normalize the entries to all absolute paths
            found_paths: list[Path] = []
            for path in data["PathHints"]:
                path_path = Path(path)
                if not path_path.is_absolute():
                    path_path = (self._model_directory / path_path).resolve()

                try:
                    if path_path.is_file():
                        found_paths.append(path_path)
                except OSError:
                    pypeline_logger.debug_log(f"Skipping missing path {path_path!s}")

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

    def add_custom_invalidation_data(
        self, pipe_id: int, configuration_hash: str, data: PipeCustomInvalidation
    ):
        with self._cursor() as cursor:
            for index, container_data in enumerate(data):
                for object_id, invalidation_blob in container_data:
                    cursor.execute(
                        "REPLACE INTO custom_dependencies VALUES (?, ?, ?, ?, ?)",
                        (
                            pipe_id,
                            configuration_hash,
                            index,
                            object_id.to_bytes(),
                            bytes(invalidation_blob),
                        ),
                    )
            self._write_metadata(cursor)

    def get_custom_invalidation_data(
        self, pipe_id: int, configuration_hash: str
    ) -> PipeCustomInvalidation:
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT argument_index, object_id, data FROM custom_dependencies"
                " WHERE pipe_id = ? AND configuration_hash = ?",
                (pipe_id, configuration_hash),
            )
            sql_result = cursor.fetchall()

        if len(sql_result) == 0:
            return []

        index_size = max(x[0] for x in sql_result)
        obj_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        result: list[list[tuple[ObjectID, bytes]]] = [[] for _ in range(index_size + 1)]
        for argument_index, object_id, data in sql_result:
            result[argument_index].append((obj_id_type.from_bytes(object_id), data))

        return result
