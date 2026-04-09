#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import tarfile
from base64 import b64decode, b64encode
from collections import defaultdict, deque
from collections.abc import Buffer, Coroutine, Generator
from contextlib import asynccontextmanager
from tempfile import SpooledTemporaryFile
from typing import IO, TYPE_CHECKING, Callable, Iterable, cast
from urllib.parse import ParseResult, urlparse

from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

import aiohttp

from revng.pypeline.utils import join_url, tar_iterate_on_members
from revng.pypeline.utils.starlette import get_middlewares

from .storage import AdditionalObjectEntry, CheckLockInvalid, CheckMetadataResult
from .storage import CustomDependency, DependencyEntry, LockCheckType, LockRequest, ObjectEntry
from .storage import RSSLockedProjectStorage, RSSProjectStorage, RSSStorage

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

# Maximum number of seconds between lock refreshes, if the client misses a
# refresh it will be considered crashed and the lock will be released
LOCK_REFRESH_INTERVAL = 10
# Maximum part size for form requests, set at 10GB
MAX_PART_SIZE = 10 * 1024 * 1024 * 1024


def _get_lock_id(request: Request) -> str:
    result = request.headers.get("x-rss-lock-id")
    assert result is not None
    return result


def _field_as_file(field: UploadFile | str) -> IO[bytes]:
    assert isinstance(field, UploadFile)
    return field.file


class _StreamingRawIOBase(io.RawIOBase):
    """
    This is a class that allows gradually emitting a byte stream as an
    asynchronous iterator. It's best suited for file formats that can be
    gradually generated (e.g. tar).
    The `generator` function takes a file-like object (this class), and each
    time it's iterated on it should generate additional bytes.
    Example usage:
    ```
    def generator(file):
        for i in range(0, 10):
            file.write(str(i).encode())
            yield

    async for chunk in _StreamingRawIOBase(generator):
        print(chunk)  # prints the bytes
    ```
    """

    def __init__(self, generator: Callable[[io.RawIOBase], Generator[None, None, None]]):
        super().__init__()
        self._queue: deque[memoryview] = deque()
        self._offset = 0
        self._generator = generator(self)
        self._closed = False

    def close(self):
        pass

    def writable(self) -> bool:
        return not self._closed

    def tell(self) -> int:
        return self._offset

    def write(self, b: Buffer):
        assert not self._closed
        view = memoryview(b)
        length = len(view)
        self._offset += length
        self._queue.append(view)
        return length

    def __aiter__(self):
        return self

    async def __anext__(self) -> memoryview:
        while len(self._queue) == 0 and not self._closed:
            # This allows the event loop to switch to another coroutine instead
            # of blocking
            await asyncio.sleep(0)

            try:
                next(self._generator)
            except StopIteration:
                self._closed = True

        if len(self._queue) == 0 and self._closed:
            raise StopAsyncIteration

        if len(self._queue) > 0:
            return self._queue.popleft()

        # In theory this code should be unreachable, but the type checker
        # doesn't know that. Raise an exception just in case.
        raise ValueError


def streaming_tar_response[T](
    items: Iterable[T],
    name_maker: Callable[[T], str],
    content_maker: Callable[[T], "ReadableBuffer"],
) -> Response:
    def generator(file: io.RawIOBase):
        with tarfile.open(fileobj=file, mode="w|") as tar:
            for element in items:
                info = tarfile.TarInfo(name=name_maker(element))
                data = content_maker(element)
                info.size = len(memoryview(data))
                info.mode = 0o644
                info.type = tarfile.REGTYPE
                tar.addfile(info, io.BytesIO(data))
                yield None

    return StreamingResponse(_StreamingRawIOBase(generator), media_type="application/x-tar")


class Endpoint:
    """
    Wrapped function that acts as the endpoint of the server, this wraps the
    following logic bits:
    * Checking if the specified lock is present
    * Adding the `project_id` parameter to the endpoint
    """

    def __init__(
        self,
        func: Callable[..., Coroutine[None, None, Response]],
        path: str,
        method: str,
        expected_initial_lock: LockCheckType,
    ):
        self._func = func
        self._path = path
        self._method = method
        self._expected_initial_lock = expected_initial_lock

    async def __call__(
        self,
        instance: RSSHTTPServer,
        request: Request,
        *args,
        **kwargs,
    ) -> Response:
        # Extract the project_id from headers, if missing return an error
        # response
        project_id = request.headers.get("x-project-id")
        if not project_id:
            return PlainTextResponse("Missing X-Project-Id header", 400)

        # If the lock type is not NONE (no lock needed), check that the lock
        # header is present and that it's valid
        if self._expected_initial_lock != LockCheckType.NONE:
            lock_id = request.headers.get("x-rss-lock-id")
            if lock_id is None:
                return PlainTextResponse("Missing x-rss-lock-id header", 400)

            try:
                async with instance.storage.get_project_storage(project_id).get_locked(
                    lock_id, self._expected_initial_lock
                ) as locked_storage:
                    return await self._func(
                        instance, request, *args, storage=locked_storage, **kwargs
                    )
            except CheckLockInvalid:
                return PlainTextResponse("Invalid or expired lock", 403)
        else:
            # Un-locked requests can refer to a project_id that is not yet
            # present in the DB, issue a creation
            unlocked_storage = instance.storage.get_project_storage(project_id)
            await unlocked_storage.create_project_if_missing()
            return await self._func(instance, request, *args, storage=unlocked_storage, **kwargs)

    def get_route(self, instance: RSSHTTPServer) -> Route:
        async def endpoint(request: Request, *args, **kwargs):
            return await self(instance, request, *args, **kwargs)

        return Route(self._path, endpoint, methods=[self._method])


def endpoint(path: str, method: str, expected_initial_lock: LockCheckType):
    """
    Decorator that allows functions in RSSHTTPServer to be registered as a
    route and automatically handled with regards to project_id and lock_id.
    Specifying an initial lock other than `None` will provide the correct
    instance of `RSSLockedProjectStorage` to the function, otherwise it will
    return the plain `RSSProjectStorage` instance.
    """

    def wrapper(func: Callable[..., Coroutine[None, None, Response]]):
        return Endpoint(func, path, method, expected_initial_lock)

    return wrapper


class RSSHTTPServer:
    def __init__(
        self,
        storage_class: type[RSSStorage],
        connection_string: str,
        notification_url: str | None,
        notification_push_psk: str | None,
        public_notification_url: str | None,
    ):
        self.storage_class = storage_class
        self.connection_string = connection_string
        # Placeholder, will be initialized by `initialize`
        self.storage: RSSStorage = cast(RSSStorage, None)

        assert ((notification_url is not None) + (notification_push_psk is not None)) in (
            0,
            2,
        ), "Either notification_url and notification_push_psk are set or neither"

        if notification_url is not None:
            self.notification_url: ParseResult | None = urlparse(notification_url)
            self.notification_push_psk = notification_push_psk
        else:
            self.notification_url = None
            self.notification_push_psk = None

        if public_notification_url is not None:
            self.public_notification_url: ParseResult | None = urlparse(public_notification_url)
        else:
            self.public_notification_url = None

    async def initialize(self):
        assert self.storage is None
        self.storage = await self.storage_class.make(self.connection_string)

    async def close(self):
        await self.storage.close()

    # Locks

    @endpoint("/lock", "POST", LockCheckType.NONE)
    async def get_lock(self, request: Request, storage: RSSProjectStorage) -> Response:
        body = await request.json()
        lock_type_str = body.get("lock_type")
        if lock_type_str == "artifact":
            lock_request = LockRequest.ARTIFACT
        elif lock_type_str == "analysis":
            lock_request = LockRequest.ANALYSIS
        else:
            return PlainTextResponse(f"Invalid lock type: {lock_type_str}", 400)

        client_version = body["version"]
        client_hash = body["pipeline_description_hash"]
        result = await storage.upgrade_project(client_version, client_hash)
        if result == CheckMetadataResult.PIPELINE_DESCRIPTION_HASH_MISSING:
            return Response(status_code=412)

        lock_id = await storage.make_lock(lock_request)
        return JSONResponse(
            {
                "lock_type": lock_type_str,
                "lock_id": lock_id,
                "refresh_interval": LOCK_REFRESH_INTERVAL,
            }
        )

    @endpoint("/renew-lock", "POST", LockCheckType.NONE)
    async def renew_lock(self, request: Request, storage: RSSProjectStorage) -> Response:
        lock_id = _get_lock_id(request)
        ok = await storage.renew_lock(lock_id)
        return Response(status_code=200 if ok else 404)

    @endpoint("/release-lock", "POST", LockCheckType.NONE)
    async def release_lock(self, request: Request, storage: RSSProjectStorage) -> Response:
        lock_id = _get_lock_id(request)
        await storage.release_lock(lock_id)
        return Response(status_code=200)

    # Savepoints

    @endpoint("/savepoint/has", "POST", LockCheckType.ARTIFACT)
    async def savepoint_has(self, request: Request, storage: RSSLockedProjectStorage) -> Response:
        body = await request.json()
        savepoint_id = body["savepoint_id"]
        container_id = body["container_id"]
        configuration_id = body["configuration_id"]
        object_ids = [b64decode(oid) for oid in body["objects"]]

        found = await storage.has_objects(savepoint_id, container_id, configuration_id, object_ids)
        return JSONResponse({"objects": [b64encode(f).decode() for f in found]})

    @endpoint("/savepoint/get", "POST", LockCheckType.ARTIFACT)
    async def savepoint_get(self, request: Request, storage: RSSLockedProjectStorage) -> Response:
        body = await request.json()
        savepoint_id = body["savepoint_id"]
        container_id = body["container_id"]
        configuration_id = body["configuration_id"]
        object_ids = [b64decode(oid) for oid in body["objects"]]

        data = await storage.get_objects(savepoint_id, container_id, configuration_id, object_ids)

        return streaming_tar_response(data, lambda e: b64encode(e[0]).decode(), lambda e: e[1])

    @endpoint("/savepoint/get-custom-invalidation-data", "GET", LockCheckType.ARTIFACT)
    async def get_custom_invalidation_data(
        self, request: Request, storage: RSSLockedProjectStorage
    ) -> Response:
        pipe_id = int(request.query_params["pipe_id"])
        configuration_hash = request.query_params["configuration_hash"]

        data = await storage.get_custom_invalidation_data(pipe_id, configuration_hash)

        return streaming_tar_response(
            data,
            lambda e: f"{e.argument_index}/{b64encode(e.object_id).decode()}",
            lambda e: e.data,
        )

    @endpoint("/savepoint/add-objects", "POST", LockCheckType.ARTIFACT)
    async def savepoint_add_objects(
        self, request: Request, storage: RSSLockedProjectStorage
    ) -> Response:
        form = await request.form(max_part_size=MAX_PART_SIZE)

        dependencies: list[DependencyEntry] = []

        dependencies_json = json.load(_field_as_file(form["dependencies"]))
        for dep in dependencies_json:
            for container_id, object_id_str, model_path in dep["dependencies"]:
                dependencies.append(
                    DependencyEntry(
                        savepoint_id_start=dep["savepoint_range"]["start"],
                        savepoint_id_end=dep["savepoint_range"]["end"],
                        configuration_hash=dep["configuration_id"],
                        container_id=container_id,
                        object_id=b64decode(object_id_str),
                        model_path=model_path,
                    )
                )

        custom_dependencies: list[CustomDependency] = []
        with tarfile.open(fileobj=_field_as_file(form["custom_invalidation"]), mode="r|") as tar:
            for member, file in tar_iterate_on_members(tar):
                # Path format: pipe_id/configuration_hash/container_index/location
                pipe_id, config_hash, container_index, location = member.name.split("/", 3)
                custom_dependencies.append(
                    CustomDependency(
                        pipe_id=int(pipe_id),
                        configuration_hash=config_hash,
                        argument_index=int(container_index),
                        object_id=b64decode(location),
                        data=file.read(),
                    )
                )

        objects: list[ObjectEntry] = []
        with tarfile.open(fileobj=_field_as_file(form["objects"]), mode="r|") as tar:
            for member, file in tar_iterate_on_members(tar):
                # Path format: savepoint_id/container_id/configuration_hash/location
                savepoint_id, container_id, config_hash, location = member.name.split("/", 3)
                objects.append(
                    ObjectEntry(
                        savepoint_id=int(savepoint_id),
                        container_id=container_id,
                        configuration_hash=config_hash,
                        object_id=b64decode(location),
                        object_id_string=member.uname + member.gname,
                        content=file.read(),
                    )
                )

        await storage.add_objects(dependencies, custom_dependencies, objects)

        return Response(status_code=200)

    # Model

    @endpoint("/model/epoch", "GET", LockCheckType.ARTIFACT)
    async def get_model_epoch(self, request: Request, storage: RSSLockedProjectStorage) -> Response:
        epoch = await storage.get_epoch()
        return JSONResponse({"epoch": epoch})

    @endpoint("/model", "GET", LockCheckType.ARTIFACT)
    async def get_model(self, request: Request, storage: RSSLockedProjectStorage) -> Response:
        model_bytes, epoch = await storage.get_model()
        model_str = model_bytes.decode() if model_bytes is not None else ""
        return JSONResponse({"epoch": epoch, "model": model_str})

    @endpoint("/model/set", "POST", LockCheckType.ANALYSIS)
    async def set_model(self, request: Request, storage: RSSLockedProjectStorage) -> Response:
        form = await request.form(max_part_size=MAX_PART_SIZE)

        invalidation_json = json.load(_field_as_file(form["invalidation"]))
        invalidation_list: list[str] = invalidation_json.get("invalidation_list", [])
        additional_objects_json = invalidation_json.get("additional_objects", [])

        additional_objects: list[AdditionalObjectEntry] = []
        for obj in additional_objects_json:
            for object_id in obj["objects"]:
                additional_objects.append(
                    AdditionalObjectEntry(
                        savepoint_id_start=obj["savepoint_range"]["start"],
                        savepoint_id_end=obj["savepoint_range"]["end"],
                        container_id=obj["container_id"],
                        configuration_hash=obj["configuration_id"],
                        object_id=object_id,
                    )
                )

        model_bytes = _field_as_file(form["model"]).read()
        result = await storage.invalidate_and_set_model(
            invalidation_list, additional_objects, model_bytes
        )

        grouped: dict[tuple[int, str, str], list[str]] = defaultdict(list)
        for row in result.invalidated:
            key = (row.savepoint_id, row.container_id, row.configuration_hash)
            grouped[key].append(row.object_id)

        invalidated_response = [
            {
                "savepoint_id": savepoint_id,
                "container_id": container_id,
                "configuration": configuration,
                "object_ids": object_ids,
            }
            for (savepoint_id, container_id, configuration), object_ids in grouped.items()
        ]

        if self.notification_url is not None:
            # Send the invalidation to the relay
            invalidation_body = {
                "type": "invalidation",
                "epoch": result.new_epoch,
                "invalidated": invalidated_response,
            }

            headers = {
                "x-project-id": request.headers["x-project-id"],
                "authorization": f"Bearer {self.notification_push_psk}",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    join_url(self.notification_url, "/publish"),
                    json=invalidation_body,
                    headers=headers,
                ) as req:
                    req.raise_for_status()

        return JSONResponse({"epoch": result.new_epoch, "invalidated": invalidated_response})

    # Metadata

    @endpoint("/metadata", "GET", LockCheckType.NONE)
    async def get_metadata(self, request: Request, storage: RSSProjectStorage) -> Response:
        metadata = await storage.get_metadata()
        return JSONResponse(metadata)

    @endpoint("/metadata/pipeline-description", "PUT", LockCheckType.NONE)
    async def put_pipeline_description(
        self, request: Request, storage: RSSProjectStorage
    ) -> Response:
        content = await request.body()
        hash_ = hashlib.sha256(content).hexdigest()
        await storage.put_pipeline_description(hash_, content)
        return PlainTextResponse(hash_)

    # File hashmap

    @endpoint("/hashmap/put-file", "POST", LockCheckType.ANY)
    async def post_hashmap_put_file(
        self, request: Request, storage: RSSLockedProjectStorage
    ) -> Response:
        result = []
        with SpooledTemporaryFile() as tempfile:
            async for chunk in request.stream():
                tempfile.write(chunk)

            tempfile.seek(0)
            with tarfile.open(fileobj=tempfile, mode="r|") as tar:
                for member, file in tar_iterate_on_members(tar):
                    content = file.read()
                    hash_ = hashlib.sha256(content).hexdigest()
                    await storage.put_file(hash_, content)
                    result.append({"name": member.name, "hash": hash_})

        return JSONResponse(result)

    @endpoint("/hashmap/get-file", "POST", LockCheckType.ANY)
    async def post_hashmap_get_file(self, request: Request, storage: RSSLockedProjectStorage):
        hashes = await request.json()
        files = await storage.get_files(hashes)
        return streaming_tar_response(files.items(), lambda e: e[0], lambda e: e[1])

    async def status(self, request: Request):
        return PlainTextResponse("OK")

    async def websocket_url(self, request: Request):
        if self.public_notification_url is None and self.notification_url is None:
            return PlainTextResponse("Notifications disabled", 400)

        if self.public_notification_url is not None:
            target_url = join_url(self.public_notification_url, "/notifications")
        else:
            assert self.notification_url is not None
            target_url = join_url(self.notification_url, "/notifications")

        target_url = re.sub("^http", "ws", target_url)
        return PlainTextResponse(target_url)

    # Return a starlette instance with all the routes
    def make_starlette(self, production: bool) -> Starlette:
        routes: list[Route] = [
            entry.get_route(self)
            for entry in vars(self.__class__).values()
            if isinstance(entry, Endpoint)
        ]
        routes.append(Route("/status", self.status, methods=["GET"]))
        routes.append(Route("/websocket-url", self.websocket_url, methods=["GET"]))

        @asynccontextmanager
        async def lifespan(app):
            await self.initialize()
            yield
            await self.close()

        return Starlette(
            debug=not production,
            routes=routes,
            middleware=get_middlewares(production, unauthenticated_paths={"/websocket-url"}),
            lifespan=lifespan,
        )
