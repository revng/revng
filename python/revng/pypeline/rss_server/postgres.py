#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import asyncio
import enum
import hashlib
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, AsyncGenerator, Callable, cast

import psycopg.sql as sql
from psycopg import AsyncCursor, Connection
from psycopg.rows import DictRow, TupleRow, dict_row
from psycopg_pool import AsyncConnectionPool

from revng.pypeline.storage.util import _OBJECTID_MAXSIZE
from revng.pypeline.utils.db_migrator import DBMigrator

from .storage import AdditionalObjectEntry, CheckLockInvalid, CheckMetadataResult
from .storage import CustomDependency, CustomDependencyEntry, DependencyEntry, InvalidatedObject
from .storage import LockCheckType, LockRequest, ModelSetResult, ObjectEntry, ProjectMetadataRow
from .storage import RSSLockedProjectStorage, RSSProjectStorage, RSSStorage

# How many seconds before a lock is considered expired
LOCK_EXPIRY_SECONDS = 30

#
# Invalidation queries
# These queries are part of the invalidation workflow, which works in 5 steps:
# 1. Create temporary tables (`INVALIDATE_CREATE_TABLES`) to store the incoming
#    data, all these tables are suffixed with a randomly-generated string
#    `<suffix>` to allow the invalidation logic to run in parallel for multiple
#    projects
# 2. Populate the tables
# 3. Run `INVALIDATE_FIND` which uses the data provided to find the objects
#    that need to be deleted
# 4. Run `INVALIDATE_DELETE_DEPENDENCIES` which removes all the dependencies of
#    the objects that are going to be deleted
# 5. Run `INVALIDATE_DELETE_OBJECTS` which actually deletes the rows from the
#    `objects` table
#

# This is a binary mask that will be used for invalidation, thanks to the
# binary structure of ObjectID, all children are guaranteed to have the parent
# prefixed, so, to check for all children the check will be
# target_object_id <= checked_object_id <= CONCAT(target_object_id, _OBJECTID_MASK)  # noqa: E800
_OBJECTID_MASK = f"'\\x{"ff" * _OBJECTID_MAXSIZE}'::bytea"

# This query creates the temporary tables needed for invalidation, these will
# hold the temporary invalidation data so that the queries below can be run
# with all the data in the database
INVALIDATE_CREATE_TABLES = """
CREATE TEMPORARY TABLE model_paths_{suffix}(
  path TEXT NOT NULL
) ON COMMIT DROP;

CREATE TEMPORARY TABLE additional_objects_{suffix}(
  savepoint_id_start   INT NOT NULL,
  savepoint_id_end     INT NOT NULL,
  container_id         TEXT NOT NULL,
  configuration_hash   TEXT NOT NULL,
  object_id            BYTEA NOT NULL
) ON COMMIT DROP;

CREATE TEMPORARY TABLE invalidated_objects_{suffix}(
  rowid tid NOT NULL
) ON COMMIT DROP;
"""

# This query looks in the `objects` table for any object that has been
# invalidated as a result of the data in `model_paths_<suffix>` and
# `additional_objects_<suffix>`. It stores the ctid of the rows in
# `invalidated_objects_<suffix>` so that the subsequent queries can go quickly
# without useless lookups.
INVALIDATE_FIND = """
INSERT INTO invalidated_objects_{suffix}
SELECT DISTINCT objects.ctid
FROM objects
JOIN (
    SELECT savepoint_id_start, savepoint_id_end,
           configuration_hash, object_id
    FROM dependencies
    JOIN model_paths_{suffix}
      ON dependencies.model_path = model_paths_{suffix}.path
    WHERE dependencies.project_id = %s
  UNION
    SELECT savepoint_id_start, savepoint_id_end,
           configuration_hash, object_id
    FROM additional_objects_{suffix}
) AS dependencies ON TRUE
WHERE objects.project_id = %s
  AND ((
        objects.object_id >= dependencies.object_id
        AND objects.object_id <= (dependencies.object_id || {objectid_mask})
    ) OR (
        dependencies.object_id != '' AND objects.object_id = ''
    )
  )
  AND objects.configuration_hash = dependencies.configuration_hash
  AND objects.savepoint_id >= dependencies.savepoint_id_start
  AND objects.savepoint_id <= dependencies.savepoint_id_end
"""

# Delete all the rows in the `dependencies` table that were connected to
# invalidated objects
INVALIDATE_DELETE_DEPENDENCIES = """
DELETE FROM dependencies
WHERE dependencies.ctid IN (
  SELECT DISTINCT dependencies.ctid FROM dependencies
  JOIN (
    SELECT savepoint_id, container_id, configuration_hash, object_id
    FROM objects
    WHERE ctid IN (SELECT rowid FROM invalidated_objects_{suffix})
  ) AS invalidated_objects ON TRUE
  WHERE dependencies.savepoint_id_start = invalidated_objects.savepoint_id
    AND dependencies.container_id = invalidated_objects.container_id
    AND dependencies.configuration_hash = invalidated_objects.configuration_hash
    AND dependencies.object_id = invalidated_objects.object_id
)
"""

# Actually delete the objects, return the identifier of the objects deleted
INVALIDATE_DELETE_OBJECTS = """
DELETE FROM objects
USING invalidated_objects_{suffix} AS inv
WHERE objects.ctid = inv.rowid
RETURNING objects.object_id_string, objects.container_id,
          objects.savepoint_id, objects.configuration_hash
"""


class LockState(enum.IntEnum):
    """
    The state that a lock can be in. The values are explicitly set since they
    will be stored in the database.
    """

    ARTIFACT_WAITING = 0
    ARTIFACT = 1
    ANALYSIS_WAITING = 2
    ANALYSIS_READING = 3
    ANALYSIS_PENDING_COMMIT = 4
    ANALYSIS_COMMITTING = 5


class AnalysisState(enum.Enum):
    NONE = enum.auto()
    READING = enum.auto()
    WRITING = enum.auto()


@dataclass
class ProjectState:
    artifact_running: bool
    analysis_state: AnalysisState


@dataclass
class LockOperationContext:
    cursor: AsyncCursor
    state: ProjectState
    should_wait_for_lock: bool = False


class CheckLockResult(enum.Enum):
    MISSING = enum.auto()
    OK = enum.auto()
    EXPIRED = enum.auto()
    TYPE_MISMATCH = enum.auto()


def _join_lockstates(*values: LockState):
    return sql.SQL(",").join((sql.Literal(int(x)) for x in values))


async def _fetch_one(cursor: AsyncCursor) -> TupleRow:
    result = await cursor.fetchone()
    assert result is not None
    return result


async def _row_generator[T](cursor: AsyncCursor, transform: Callable[[TupleRow], T]) -> list[T]:
    result: list[T] = []
    async for row in cursor:
        result.append(transform(row))
    return result


class Migrator(DBMigrator):
    def __init__(self, conninfo: str):
        super().__init__(
            "pypeline", PurePath(self.__module__.replace(".", "/")).parent / "postgres_migrations"
        )
        self._connection = Connection.connect(conninfo)
        self._app_name = "rss_server"

    def _create_tables_if_missing(self):
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS "
            "migrations(app_name TEXT PRIMARY KEY, version INT NOT NULL)"
        )
        self._connection.commit()

    def _get_last_migration(self) -> int:
        result = self._connection.execute(
            "SELECT version FROM migrations WHERE app_name = %s", (self._app_name,)
        ).fetchone()
        return 0 if result is None else result[0]

    def _apply_migration(self, version: int, body: str):
        self._connection.execute(body.encode())
        self._connection.execute(
            "INSERT INTO migrations VALUES (%s, %s) "
            "ON CONFLICT (app_name) DO UPDATE SET version = excluded.version",
            (self._app_name, version),
        )
        self._connection.commit()

    def truncate_locks(self):
        self._connection.execute("TRUNCATE TABLE locks")
        self._connection.commit()


class PostgresRSSStorage(RSSStorage):
    """RSSStorage backed by PostgreSQL via psycopg 3"""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    @classmethod
    async def make(cls, connection_string: str) -> PostgresRSSStorage:
        pool = AsyncConnectionPool(connection_string, min_size=0, max_size=1000, open=False)
        await pool.open()
        return cls(pool)

    async def close(self):
        await self.pool.close()

    @staticmethod
    def migrate(connection_string: str):
        migrator = Migrator(connection_string)
        migrator.migrate()
        migrator.truncate_locks()

    @classmethod
    def background_tasks(cls, connection_string: str):
        async def cleanup():
            instance = await cls.make(connection_string)
            return await instance._cleanup_locks()

        return [asyncio.create_task(cleanup())]

    @asynccontextmanager
    async def _connection(self):
        async with self.pool.connection() as conn:
            yield conn

    @asynccontextmanager
    async def _cursor(self):
        async with self._connection() as conn:
            async with conn.transaction():
                yield conn.cursor()

    async def _cleanup_locks(self):
        while True:
            await asyncio.sleep(30)
            async with self._cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM locks WHERE refresh_timestamp + make_interval(secs => %s) < NOW()"
                    " AND in_use is FALSE"
                    " RETURNING project_id, lock_id",
                    (LOCK_EXPIRY_SECONDS,),
                )

                already_unblocked_projects: set[str] = set()
                async for project_id, lock_id in cursor:
                    if project_id in already_unblocked_projects:
                        continue

                    project_storage = self.get_project_storage(project_id)
                    async with project_storage._mutate_project_lock(None) as context:
                        await project_storage._unlock_next(context.cursor, context.state)

                    already_unblocked_projects.add(project_id)

    def get_project_storage(self, project_id: str) -> PostgresRSSProjectStorage:
        return PostgresRSSProjectStorage(self, project_id)


class PostgresRSSProjectStorage(RSSProjectStorage):
    """
    This implements the `RSSProjectStorage` interface, allowing to interact
    with a single project instance. This class mainly deals with locking and
    managing metadata. For actual storage operation the lock must be held,
    which happens via the use of `get_locked` to get an instance of
    `RSSLockedProjectStorage`.
    """

    def __init__(self, parent: PostgresRSSStorage, project_id: str):
        self._parent = parent
        self._project_id = project_id

    def _connection(self):
        return self._parent._connection()

    def _cursor(self):
        return self._parent._cursor()

    #
    # Locking endpoints
    #

    async def _compute_project_state(self, cursor: AsyncCursor) -> ProjectState:
        await cursor.execute(
            sql.SQL(
                "SELECT lock_type FROM locks WHERE project_id = %s "
                "AND lock_type IN ({values}) LIMIT 1"
            ).format(
                values=_join_lockstates(
                    LockState.ANALYSIS_READING,
                    LockState.ANALYSIS_PENDING_COMMIT,
                    LockState.ANALYSIS_COMMITTING,
                )
            ),
            (self._project_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            analysis_state = AnalysisState.NONE
        else:
            lt = LockState(row[0])
            if lt == LockState.ANALYSIS_READING:
                analysis_state = AnalysisState.READING
            else:
                analysis_state = AnalysisState.WRITING

        await cursor.execute(
            "SELECT 1 FROM locks WHERE project_id = %s AND lock_type = %s LIMIT 1",
            (self._project_id, int(LockState.ARTIFACT)),
        )
        artifact_running = (await cursor.fetchone()) is not None

        return ProjectState(
            artifact_running=artifact_running,
            analysis_state=analysis_state,
        )

    @asynccontextmanager
    async def _mutate_project_lock(self, lock_id: str | None):
        # TODO: the pg_advisory_xact_lock is global w.r.t. the postgres
        # instance, if more places need to use them then the bits used from the
        # hash should be reduced (e.g. 60 bits) and the unused bits would be
        # used to encode the "lock type"
        digest = hashlib.sha256(self._project_id.encode()).digest()
        project_key = int.from_bytes(digest[:8], "little", signed=True)

        async with self._connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await cursor.execute("SELECT pg_advisory_xact_lock(%s)", (project_key,))
                    # TODO: the project state is always computed but the `body`
                    # might not read it at all. In the future `ProjectState`
                    # could compute the data lazily.
                    state = await self._compute_project_state(cursor)
                    context = LockOperationContext(cursor, state)
                    yield context
                    if not context.should_wait_for_lock:
                        # We don't have to wait, return immediately
                        return

                    # If here there lock has to be waited, run `LISTEN` on the
                    # requested channel while the lock is still being held
                    assert lock_id is not None
                    channel = sql.Identifier(f"lock_{self._project_id}_{lock_id}")
                    await cursor.execute(sql.SQL("LISTEN {}").format(channel))

            # When here the transaction has been committed, so the
            # `pg_advisory_xact_lock` has been closed
            async for _ in connection.notifies():
                break
            await connection.execute(sql.SQL("UNLISTEN {}").format(channel))

    @asynccontextmanager
    async def _upgrade_to_commiting_lock(self, lock_id: str):
        """
        Temporarily upgrade an `ANALYSIS_READING` lock to `ANALYSIS_COMMITTING`
        lock, waiting for pending `ARTIFACT` locks to finish before returning.
        """

        async with self._mutate_project_lock(lock_id) as context:
            if not context.state.artifact_running:
                # If there aren't any artifact locks reading the model then the
                # lock can be directly upgraded
                lock_state = LockState.ANALYSIS_COMMITTING
            else:
                # If here there are artifact locks active. Set the lock to
                # `ANALYSIS_PENDING_COMMIT` and wait to be promoted
                lock_state = LockState.ANALYSIS_PENDING_COMMIT
                context.should_wait_for_lock = True

            await context.cursor.execute(
                "UPDATE locks SET lock_type = %s WHERE project_id = %s AND lock_id = %s",
                (int(lock_state), self._project_id, lock_id),
            )

        try:
            yield None
        finally:
            async with self._mutate_project_lock(lock_id) as context:
                # Downgrade the lock back to `ANALYSIS_READING`
                await context.cursor.execute(
                    "UPDATE locks SET lock_type = %s WHERE project_id = %s AND lock_id = %s",
                    (int(LockState.ANALYSIS_READING), self._project_id, lock_id),
                )

    async def make_lock(self, lock_type: LockRequest) -> str:
        lock_id = str(uuid.uuid4())

        async with self._mutate_project_lock(lock_id) as context:
            if lock_type == LockRequest.ARTIFACT:
                if context.state.analysis_state != AnalysisState.WRITING:
                    lock_state = LockState.ARTIFACT
                else:
                    lock_state = LockState.ARTIFACT_WAITING
                    context.should_wait_for_lock = True

            elif lock_type == LockRequest.ANALYSIS:
                if context.state.analysis_state == AnalysisState.NONE:
                    lock_state = LockState.ANALYSIS_READING
                else:
                    lock_state = LockState.ANALYSIS_WAITING
                    context.should_wait_for_lock = True

            else:
                raise ValueError

            await context.cursor.execute(
                "INSERT INTO locks (project_id, lock_id, lock_type) VALUES (%s, %s, %s)",
                (self._project_id, lock_id, int(lock_state)),
            )

        return lock_id

    async def _unlock_next(self, cursor: AsyncCursor, state: ProjectState):
        pending_locks = await self._get_pending_locks(cursor, self._project_id)

        to_promote: list[tuple[str, LockState]] = []
        for lock_id, lock_state in pending_locks:
            if state.analysis_state == AnalysisState.WRITING and state.artifact_running:
                break

            if lock_state == LockState.ARTIFACT_WAITING:
                if state.analysis_state != AnalysisState.WRITING:
                    to_promote.append((lock_id, LockState.ARTIFACT))
                    state.artifact_running = True

            elif lock_state == LockState.ANALYSIS_WAITING:
                if state.analysis_state == AnalysisState.NONE:
                    to_promote.append((lock_id, LockState.ANALYSIS_READING))
                    state.analysis_state = AnalysisState.READING

            elif lock_state == LockState.ANALYSIS_PENDING_COMMIT:  # noqa: SIM102
                if not state.artifact_running:
                    to_promote.append((lock_id, LockState.ANALYSIS_COMMITTING))
                    state.analysis_state = AnalysisState.WRITING

        for lock_id, new_state in to_promote:
            await cursor.execute(
                "UPDATE locks SET lock_type = %s WHERE project_id = %s AND lock_id = %s",
                (int(new_state), self._project_id, lock_id),
            )
            await cursor.execute(
                sql.SQL("NOTIFY {}").format(sql.Identifier(f"lock_{self._project_id}_{lock_id}"))
            )

    async def release_lock(self, lock_id: str):
        async with self._mutate_project_lock(lock_id) as context:
            await context.cursor.execute(
                "SELECT in_use FROM locks WHERE project_id = %s AND lock_id = %s",
                (self._project_id, lock_id),
            )
            result = await context.cursor.fetchone()
            if result is None:
                raise ValueError("Missing lock")
            if result[0]:
                raise ValueError("Lock is in use")

            await context.cursor.execute(
                "DELETE FROM locks WHERE project_id = %s AND lock_id = %s",
                (self._project_id, lock_id),
            )
            await self._unlock_next(context.cursor, context.state)

    async def _is_lock_compatible(
        self, cursor: AsyncCursor, project_id: str, lock_id: str, type_: LockCheckType
    ) -> CheckLockResult:
        if type_ == LockCheckType.NONE:
            return CheckLockResult.OK

        await cursor.execute(
            "SELECT refresh_timestamp + make_interval(secs => %s) > NOW(), lock_type"
            " FROM locks WHERE project_id = %s AND lock_id = %s",
            (LOCK_EXPIRY_SECONDS, project_id, lock_id),
        )
        result = await cursor.fetchone()
        if result is None:
            return CheckLockResult.MISSING
        if not result[0]:
            return CheckLockResult.EXPIRED

        actual_lock_type = result[1]
        if type_ == LockCheckType.ANY:
            return CheckLockResult.OK
        elif type_ == LockCheckType.ARTIFACT:
            ok = actual_lock_type in (int(LockState.ARTIFACT), int(LockState.ANALYSIS_READING))
        elif type_ == LockCheckType.ANALYSIS:
            ok = actual_lock_type == int(LockState.ANALYSIS_READING)

        return CheckLockResult.OK if ok else CheckLockResult.TYPE_MISMATCH

    async def renew_lock(self, lock_id: str):
        async with self._cursor() as cursor:
            state = await self._is_lock_compatible(
                cursor, self._project_id, lock_id, LockCheckType.ANY
            )
            if state == CheckLockResult.MISSING:
                return False
            elif state == CheckLockResult.EXPIRED:
                await self.release_lock(lock_id)
                return False

            await cursor.execute(
                "UPDATE locks SET refresh_timestamp = NOW() WHERE project_id = %s AND lock_id = %s",
                (self._project_id, lock_id),
            )
            return True

    async def _get_pending_locks(
        self, cursor: AsyncCursor, project_id: str
    ) -> list[tuple[str, LockState]]:
        query = sql.SQL(
            "SELECT lock_id, lock_type FROM locks WHERE "
            "project_id = %s AND lock_type IN ({values}) ORDER BY creation_timestamp ASC"
        ).format(
            values=_join_lockstates(
                LockState.ARTIFACT_WAITING,
                LockState.ANALYSIS_WAITING,
                LockState.ANALYSIS_PENDING_COMMIT,
            )
        )

        await cursor.execute(query, (project_id,))
        return await _row_generator(cursor, lambda r: (r[0], LockState(r[1])))

    @asynccontextmanager
    async def get_locked(
        self, lock_id: str, initial_type: LockCheckType
    ) -> AsyncGenerator[PostgresRSSLockedProjectStorage]:
        # A raw connection (as opposed to a cursor with a transaction) needs to
        # be acquired because this function manages commits manually
        async with self._connection() as connection:
            # Actually check that the lock is valid in the DB w.r.t. the type
            # provided
            async with connection.cursor() as cursor:
                check_lock_result = await self._is_lock_compatible(
                    cursor, self._project_id, lock_id, initial_type
                )

            # Since this is a context manager the only way to exit is to throw
            # an exception in case the lock turned out to be invalid.
            if check_lock_result != CheckLockResult.OK:
                raise CheckLockInvalid

            # Set the `in_use` flag of the lock, this prevents it from being deleted
            await connection.execute(
                "UPDATE locks SET in_use = TRUE WHERE project_id = %s AND lock_id = %s",
                (self._project_id, lock_id),
            )
            await connection.commit()

            # Yield, finishing the enter portion of the context manager
            try:
                async with connection.transaction():
                    async with connection.cursor() as cursor:
                        yield PostgresRSSLockedProjectStorage(
                            self, cursor, self._project_id, lock_id
                        )
            finally:
                # Here the context manager is exiting, unset the `in_use` flag,
                # so that the lock can be deleted if needed
                await connection.execute(
                    "UPDATE locks SET in_use = FALSE WHERE project_id = %s AND lock_id = %s",
                    (self._project_id, lock_id),
                )
                await connection.commit()

    #
    # Metadata
    #

    async def get_metadata(self) -> ProjectMetadataRow:
        async with self._cursor() as cursor:
            await cursor.execute(
                "SELECT pipeline_description_hash, version, "
                "extract(epoch FROM last_fetch)::float AS last_fetch, "
                "extract(epoch FROM last_object_save)::float AS last_object_save, "
                "extract(epoch FROM last_model_save)::float AS last_model_save "
                "FROM project WHERE project_id = %s",
                (self._project_id,),
            )

            cursor.row_factory = cast(Any, dict_row)
            row = cast(DictRow, await _fetch_one(cursor))
            return cast(ProjectMetadataRow, row)

    async def put_pipeline_description(self, hash_: str, content: bytes):
        async with self._cursor() as cursor:
            await cursor.execute(
                "INSERT INTO pipeline_descriptions (hash, content) VALUES (%s, %s)"
                "ON CONFLICT (hash) DO NOTHING",
                (hash_, content),
            )

    #
    # Project lifecycle
    #

    async def create_project_if_missing(self):
        """
        Create a project entry in the project table. If it already exists this
        function does nothing.
        """
        async with self._cursor() as cursor:
            await cursor.execute(
                "INSERT INTO project (project_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (self._project_id,),
            )

    async def upgrade_project(
        self, version: str, pipeline_description_hash: str
    ) -> CheckMetadataResult:
        async with self._cursor() as cursor:
            await cursor.execute(
                "SELECT version, pipeline_description_hash FROM project WHERE project_id = %s",
                (self._project_id,),
            )
            result = await _fetch_one(cursor)

            if result[0] == version and result[1] == pipeline_description_hash:
                return CheckMetadataResult.OK

            # Check that the pipeline description hash is present, otherwise do
            # an early return
            if result[1] != pipeline_description_hash:
                await cursor.execute(
                    "SELECT 1 FROM pipeline_descriptions WHERE hash = %s",
                    (pipeline_description_hash,),
                )
                if (await cursor.fetchone()) is None:
                    return CheckMetadataResult.PIPELINE_DESCRIPTION_HASH_MISSING

        # If here everything needs to be pruned, first obtain an
        # analysis lock in the committing state
        lock_id = await self.make_lock(LockRequest.ANALYSIS)
        try:
            async with self.get_locked(lock_id, LockCheckType.ANALYSIS) as locked_storage:
                async with self._upgrade_to_commiting_lock(lock_id):
                    await locked_storage.prune_objects()
                    await cursor.execute(
                        "UPDATE project SET version = %s, pipeline_description_hash = %s, "
                        "last_object_save = NOW() WHERE project_id = %s",
                        (version, pipeline_description_hash, self._project_id),
                    )
        finally:
            await self.release_lock(lock_id)

        return CheckMetadataResult.PRUNE_DONE


class PostgresRSSLockedProjectStorage(RSSLockedProjectStorage):
    """
    This class is an implementation of `RSSLockedProjectStorage`, which is
    guaranteed to have the lock being held as long as this class is used.
    """

    def __init__(
        self, parent: PostgresRSSProjectStorage, cursor: AsyncCursor, project_id: str, lock_id: str
    ):
        self._parent = parent
        self._cursor = cursor
        self._project_id = project_id
        self._lock_id = lock_id

    #
    # Pypeline-facing endpoints
    #

    async def has_objects(
        self,
        savepoint_id: int,
        container_id: str,
        configuration_hash: str,
        object_ids: list[bytes],
    ) -> list[bytes]:
        if len(object_ids) == 0:
            return []

        await self._cursor.execute(
            "SELECT object_id FROM objects WHERE project_id = %s AND savepoint_id = %s "
            "AND container_id = %s AND configuration_hash = %s AND object_id = ANY(%s)",
            (self._project_id, savepoint_id, container_id, configuration_hash, object_ids),
        )
        return await _row_generator(self._cursor, lambda r: r[0])

    async def get_objects(
        self,
        savepoint_id: int,
        container_id: str,
        configuration_hash: str,
        object_ids: list[bytes],
    ) -> list[tuple[bytes, bytes]]:
        if len(object_ids) == 0:
            return []

        await self._cursor.execute(
            "SELECT object_id, content FROM objects "
            "WHERE project_id = %s AND savepoint_id = %s AND container_id = %s "
            "AND configuration_hash = %s AND object_id = ANY(%s)",
            (self._project_id, savepoint_id, container_id, configuration_hash, object_ids),
        )
        result = await _row_generator(self._cursor, lambda r: (r[0], r[1]))
        await self._cursor.execute(
            "UPDATE project SET last_fetch = NOW() WHERE project_id = %s",
            (self._project_id,),
        )
        return result

    async def get_custom_invalidation_data(
        self,
        pipe_id: int,
        configuration_hash: str,
    ) -> list[CustomDependencyEntry]:
        await self._cursor.execute(
            "SELECT argument_index, object_id, data FROM custom_dependencies "
            "WHERE project_id = %s AND pipe_id = %s AND configuration_hash = %s",
            (self._project_id, pipe_id, configuration_hash),
        )
        return await _row_generator(self._cursor, lambda r: CustomDependencyEntry(r[0], r[1], r[2]))

    async def _run_invalidation(
        self,
        model_paths: list[str],
        additional_objects: list[AdditionalObjectEntry],
    ) -> list[InvalidatedObject]:
        if len(model_paths) == 0 and len(additional_objects) == 0:
            return []

        suffix = uuid.uuid4().hex

        # Step 1, create the temporary tables
        await self._cursor.execute(INVALIDATE_CREATE_TABLES.format(suffix=suffix).encode())

        # Step 2, insert the incoming data
        if len(model_paths) > 0:
            await self._cursor.executemany(
                f"INSERT INTO model_paths_{suffix} VALUES (%s)".encode(),
                [(p,) for p in model_paths],
            )

        if len(additional_objects) > 0:
            await self._cursor.executemany(
                f"INSERT INTO additional_objects_{suffix} VALUES (%s, %s, %s, %s, %s)".encode(),
                (
                    (
                        ao.savepoint_id_start,
                        ao.savepoint_id_end,
                        ao.container_id,
                        ao.configuration_hash,
                        ao.object_id,
                    )
                    for ao in additional_objects
                ),
            )

        # Step 3, find the objects to delete, store them into
        # `invalidated_objects_<suffix>`
        await self._cursor.execute(
            INVALIDATE_FIND.format(suffix=suffix, objectid_mask=_OBJECTID_MASK).encode(),
            (self._project_id, self._project_id),
        )

        # Step 4, delete the dependencies for all the objects that are going to
        # be deleted
        await self._cursor.execute(INVALIDATE_DELETE_DEPENDENCIES.format(suffix=suffix).encode())

        # Step 5, actually delete the objects and return the invalidated data
        await self._cursor.execute(INVALIDATE_DELETE_OBJECTS.format(suffix=suffix).encode())
        return await _row_generator(
            self._cursor,
            lambda r: InvalidatedObject(
                object_id=r[0],
                container_id=r[1],
                savepoint_id=r[2],
                configuration_hash=r[3],
            ),
        )

    async def prune_objects(self):
        await self._cursor.execute("DELETE FROM objects WHERE project_id = %s", (self._project_id,))
        await self._cursor.execute(
            "DELETE FROM dependencies WHERE project_id = %s", (self._project_id,)
        )
        await self._cursor.execute(
            "DELETE FROM custom_dependencies WHERE project_id = %s", (self._project_id,)
        )

    async def get_epoch(self) -> int:
        await self._cursor.execute(
            "SELECT epoch FROM project WHERE project_id = %s", (self._project_id,)
        )
        row = await self._cursor.fetchone()
        assert row is not None
        return row[0]

    async def get_model(self) -> tuple[bytes | None, int]:
        await self._cursor.execute(
            "SELECT model, epoch FROM project WHERE project_id = %s", (self._project_id,)
        )
        row = await self._cursor.fetchone()
        assert row is not None
        model = bytes(row[0]) if row[0] is not None else None
        return (model, row[1])

    async def add_objects(
        self,
        dependencies: list[DependencyEntry],
        custom_dependencies: list[CustomDependency],
        objects: list[ObjectEntry],
    ):
        if len(dependencies) > 0:
            await self._cursor.executemany(
                "INSERT INTO dependencies "
                "(project_id, savepoint_id_start, savepoint_id_end, container_id, "
                " configuration_hash, object_id, model_path)"
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                (
                    (
                        self._project_id,
                        d.savepoint_id_start,
                        d.savepoint_id_end,
                        d.container_id,
                        d.configuration_hash,
                        d.object_id,
                        d.model_path,
                    )
                    for d in dependencies
                ),
            )

        if custom_dependencies:
            await self._cursor.executemany(
                "INSERT INTO custom_dependencies"
                "(project_id, pipe_id, configuration_hash, argument_index, object_id, data) "
                "VALUES (%s, %s, %s, %s, %s, %s)"
                "ON CONFLICT (project_id, pipe_id, configuration_hash, argument_index, "
                "  object_id) DO UPDATE SET data = EXCLUDED.data",
                (
                    (
                        self._project_id,
                        cd.pipe_id,
                        cd.configuration_hash,
                        cd.argument_index,
                        cd.object_id,
                        cd.data,
                    )
                    for cd in custom_dependencies
                ),
            )

        await self._cursor.executemany(
            "INSERT INTO objects "
            "(project_id, savepoint_id, container_id, configuration_hash, object_id, "
            "object_id_string, content) VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (project_id, savepoint_id, container_id, configuration_hash, "
            " object_id) DO UPDATE SET content = EXCLUDED.content",
            (
                (
                    self._project_id,
                    o.savepoint_id,
                    o.container_id,
                    o.configuration_hash,
                    o.object_id,
                    o.object_id_string,
                    o.content,
                )
                for o in objects
            ),
        )

        await self._cursor.execute(
            "UPDATE project SET last_object_save = NOW() WHERE project_id = %s",
            (self._project_id,),
        )

    async def invalidate_and_set_model(
        self,
        model_paths: list[str],
        additional_objects: list[AdditionalObjectEntry],
        model_bytes: bytes,
    ) -> ModelSetResult:
        async with self._parent._upgrade_to_commiting_lock(self._lock_id):
            invalidated_rows = await self._run_invalidation(model_paths, additional_objects)

            await self._cursor.execute(
                "UPDATE project SET model = %s, epoch = epoch + 1, last_model_save = NOW()"
                " WHERE project_id = %s RETURNING epoch",
                (model_bytes, self._project_id),
            )
            row = await self._cursor.fetchone()
            assert row is not None

        return ModelSetResult(new_epoch=row[0], invalidated=invalidated_rows)

    async def put_file(self, hash_: str, content: bytes):
        # Check if the file has already been uploaded, this saves us from
        # shipping the (possibly big) binary over to the DB
        await self._cursor.execute(
            "SELECT 1 FROM file_storage WHERE project_id = %s and hash = %s",
            (self._project_id, hash_),
        )
        if (await self._cursor.fetchone()) is not None:
            return

        await self._cursor.execute(
            "INSERT INTO file_storage (project_id, hash, content) VALUES (%s, %s, %s)"
            "ON CONFLICT (project_id, hash) DO NOTHING",
            (self._project_id, hash_, content),
        )

    async def get_files(self, hashes: list[str]) -> dict[str, bytes]:
        if len(hashes) == 0:
            return {}

        await self._cursor.execute(
            "SELECT hash, content FROM file_storage WHERE project_id = %s AND hash = ANY(%s)",
            (self._project_id, hashes),
        )
        result = {}
        async for row in self._cursor:
            result[row[0]] = bytes(row[1])
        return result
