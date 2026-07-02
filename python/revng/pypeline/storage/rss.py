#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import hashlib
import io
import json
import os
import tarfile
import threading
import time
from base64 import b64decode, b64encode
from collections import defaultdict
from collections.abc import AsyncGenerator, Buffer, Collection, Iterable, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import SpooledTemporaryFile
from urllib.parse import ParseResult, parse_qsl, urlparse

import requests
import yaml
from aiohttp import ClientSession
from aiohttp.client import ClientTimeout
from aiohttp.client_exceptions import ClientError

from revng.pypeline import __version__ as version
from revng.pypeline.model import Model, ModelPathSet
from revng.pypeline.object import ObjectID
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.task.pipe import PipeCustomInvalidation
from revng.pypeline.utils import PypelineException, join_url, tar_iterate_on_members
from revng.pypeline.utils.buffered_reader import BufferedReader
from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.pipeline import get_pipeline_description
from revng.pypeline.utils.registry import get_singleton

from .file_provider import FileRequest
from .storage_provider import ContainerLocation, FileStorageEntry, LockType, ObjectsToInvalidate
from .storage_provider import PipeDependencies, ProjectID, ProjectMetadata, SetModelResult
from .storage_provider import StorageProvider, StorageProviderFactory


@dataclass
class _LockEntry:
    headers: dict[str, str]
    refresh_interval: float
    next_renewal: float


class _LockRenewalThread(threading.Thread):
    """Daemon thread that periodically renews all registered RSS locks."""

    # How long to anticipate the refresh, in (0, 1). e.g. 0.2 will mean that
    # the refresh will happen at 80% expired time
    REFRESH_MARGIN = 0.2

    def __init__(self, renew_url: str):
        super().__init__(daemon=True)
        self.entries_lock = threading.Lock()
        self.entries: dict[str, _LockEntry] = {}
        self.event = threading.Event()
        self.url = renew_url
        self.start()

    @contextmanager
    def track_lock(self, lock_id: str, headers: dict[str, str], refresh_interval: float):
        """
        This context manager takes care of registering the lock for refreshing
        while entering and removes it on exit. Do note that the actual refresh
        is done in a separate thread
        """

        actual_refresh_interval = refresh_interval * (1 - self.__class__.REFRESH_MARGIN)
        with self.entries_lock:
            self.entries[lock_id] = _LockEntry(
                headers=headers,
                refresh_interval=actual_refresh_interval,
                next_renewal=time.monotonic() + actual_refresh_interval,
            )
            self.event.set()
        pypeline_logger.debug_log(f"Starting refresh for lock {lock_id}")

        try:
            yield None
        finally:
            with self.entries_lock:
                self.entries.pop(lock_id, None)
            pypeline_logger.debug_log(f"Stopping refresh for lock {lock_id}")

    def run(self):
        next_wait_seconds = 1.0
        while True:
            self.event.wait(next_wait_seconds)
            self.event.clear()
            self.refresh_locks()

            with self.entries_lock:
                entries_copy = self.entries.copy()
            if len(entries_copy) > 0:
                global_next_renewal = min(e.next_renewal for e in entries_copy.values())
                next_wait_seconds = max(global_next_renewal - time.monotonic(), 1.0)
            else:
                next_wait_seconds = 1.0

    def refresh_locks(self):
        with self.entries_lock:
            entries_copy = self.entries.copy()

        now = time.monotonic()
        locks_to_bump: set[str] = set()
        locks_to_drop: set[str] = set()
        for lock_id, entry in entries_copy.items():
            if entry.next_renewal > now:
                continue

            pypeline_logger.debug_log(f"Refreshing lock {lock_id}")
            headers = {**entry.headers, "X-RSS-Lock-ID": lock_id}
            response = requests.post(self.url, headers=headers, timeout=1.0)
            if not response.ok:
                pypeline_logger.debug_log(f"lock refresh failed for {lock_id}, removing lock")
                locks_to_drop.add(lock_id)
            else:
                locks_to_bump.add(lock_id)

        with self.entries_lock:
            for lock_id in locks_to_drop:
                self.entries.pop(lock_id, None)

            for lock_id in locks_to_bump:
                if lock_id not in self.entries:
                    continue

                next_renewal = now + self.entries[lock_id].refresh_interval
                self.entries[lock_id].next_renewal = next_renewal


class RSSStorageProviderFactory(StorageProviderFactory):
    def __init__(self, url: str):
        """
        This StorageProviderFactory uses the following scheme:
        rss://<ip>:<port>/?proto=http(s)
        The proto is to indicate if communication needs to happen in HTTP or
        HTTPS. By default HTTP is used if not specified.
        """
        parsed = urlparse(url)
        query_parameters = dict(parse_qsl(parsed.query))
        # Get the underlying scheme from the `?proto=` parameter
        scheme = query_parameters.get("proto", "http")
        assert scheme in ("http", "https")
        self._base_url = parsed._replace(scheme=scheme, query="")

        # Lazily initialize renewal_thread, only do it on the first call to
        # `get`. While there is a working asyncio event loop, a lot of the
        # underlying methods are sync (e.g. Pipeline), this would block the
        # event loop for a long time and prevent locks being refreshed in time.
        self._renewal_thread: _LockRenewalThread | None = None

    @classmethod
    def scheme(cls) -> str:
        return "rss"

    def _join_url(self, path: str) -> str:
        return join_url(self._base_url, path)

    async def _acquire_lock(
        self, session: ClientSession, lock_type: str, pipeline_description: bytes
    ):
        body = {
            "lock_type": lock_type,
            "version": version,
            "pipeline_description_hash": hashlib.sha256(pipeline_description).hexdigest(),
        }

        async with session.post(self._join_url("/lock"), json=body) as response:
            if response.status == 200:
                return await response.json()
            elif response.status != 412:
                response.raise_for_status()

        # If here the pipeline description needs to be uploaded
        async with session.put(
            self._join_url("/metadata/pipeline-description"), data=pipeline_description
        ) as response:
            response.raise_for_status()

        # Re-acquire the lock
        async with session.post(self._join_url("/lock"), json=body) as response:
            response.raise_for_status()
            return await response.json()

    @asynccontextmanager
    async def _with_lock(
        self,
        lock_type: str,
        pipeline_description: bytes,
        session: ClientSession,
        headers: dict[str, str],
    ):
        # Acquire the lock
        lock_data = await self._acquire_lock(session, lock_type, pipeline_description)
        lock_id = lock_data["lock_id"]
        refresh_interval = lock_data["refresh_interval"]

        # Lazily initialize the renewal thread and add the lock to it
        if self._renewal_thread is None:
            self._renewal_thread = _LockRenewalThread(self._join_url("/renew-lock"))

        try:
            with self._renewal_thread.track_lock(lock_id, headers, refresh_interval):
                yield lock_id
        finally:
            # Release the lock
            try:
                async with session.post(
                    self._join_url("/release-lock"),
                    headers={"X-RSS-Lock-ID": lock_id},
                    timeout=ClientTimeout(total=3.0),
                ) as response:
                    response.raise_for_status()
            except ClientError as e:
                pypeline_logger.log(f"Exception while releasing the lock: {str(e)}")

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
        assert project_id is not None

        headers: dict[str, str] = {}
        headers["X-Project-Id"] = project_id
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"

        lock_type_str = "artifact" if lock_type == LockType.ARTIFACT else "analysis"
        pipeline_description = yaml.safe_dump(get_pipeline_description(pipeline)).encode()

        async with ClientSession(headers=headers) as session:
            async with self._with_lock(
                lock_type_str, pipeline_description, session, headers
            ) as lock_id:
                provider_headers = {**headers, "X-RSS-Lock-ID": lock_id}
                provider = RSSStorageProvider(
                    base_url=self._base_url,
                    headers=provider_headers,
                    pipeline_description=pipeline_description,
                )
                yield provider

    def get_notification_websocket(self) -> str:
        req = requests.get(self._join_url("/websocket-url"))
        return req.text


class SpooledTarWriter:
    """
    This class wraps a tar file within a SpooledTemporaryFile, useful for
    situations where the tar might be big and needs to be spooled to disk.
    Files are added via `add_file` and the resulting file can be retrieved
    with `get_file`. Once the file is retrieved new files cannot be added to
    the tar.
    """

    def __init__(self):
        self._file = SpooledTemporaryFile(max_size=2 * 1024 * 1024)
        self._tar: tarfile.TarFile | None = tarfile.open(fileobj=self._file, mode="w|")

    def add_file(self, name: str, content: Buffer | io.IOBase, uname: str = "", gname: str = ""):
        assert self._tar is not None

        if isinstance(content, Buffer):
            content = io.BytesIO(content)

        original_position = content.tell()
        info = tarfile.TarInfo(name=name)
        info.size = content.seek(0, os.SEEK_END)
        info.mode = 0o644
        info.type = tarfile.REGTYPE
        info.uname = uname
        info.gname = gname

        content.seek(original_position, os.SEEK_SET)
        self._tar.addfile(info, content)

    def get_file(self) -> io.IOBase:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

        self._file.seek(0)
        return self._file


class RSSClientException(PypelineException):
    def __init__(self, response: requests.Response):
        assert 400 <= response.status_code < 600
        text = response.text
        super().__init__(f"RSS request failed with status {response.status_code}: {text}")
        self.status_code = response.status_code
        self.text = text


_FilesType = Mapping[str, tuple[str, io.IOBase | str | bytes]]


def _object_id_to_str(object_id: ObjectID):
    return b64encode(object_id.to_bytes()).decode()


def _object_id_from_str(type_: type[ObjectID], string: str):
    return type_.from_bytes(b64decode(string))


def _object_id_serialize_split(object_id: ObjectID) -> tuple[str, str]:
    # Serialize an object ID as a pair of strings, the first 31-characters long
    # and the second unbound. This is needed to stuff the serialized object_id
    # into the `uname` and `gname` of a tar file without using additional
    # headers. If the gname is longer than 31 characters it will get sent to a
    # pax header anyways.
    string = object_id.serialize()
    return (string[:31], string[31:])


class RSSStorageProvider(StorageProvider):
    def __init__(
        self,
        base_url: ParseResult,
        headers: dict[str, str],
        pipeline_description: bytes,
    ):
        self._base_url = base_url

        self._session = requests.Session()
        self._session.hooks["response"].append(self._response_hook)
        self._session.headers.update(headers)

    def _join_url(self, path: str) -> str:
        return join_url(self._base_url, path)

    @staticmethod
    def _response_hook(response: requests.Response, *args, **kwargs):
        if 400 <= response.status_code < 600:
            raise RSSClientException(response)

    def has(
        self,
        location: ContainerLocation,
        keys: Collection[ObjectID],
    ) -> Iterable[ObjectID]:
        body = {
            "savepoint_id": location.savepoint_id,
            "container_id": location.container_id,
            "configuration_id": location.configuration_id,
            "objects": [_object_id_to_str(obj) for obj in keys],
        }
        response = self._session.post(self._join_url("/savepoint/has"), json=body)

        data = response.json()
        obj_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        return [_object_id_from_str(obj_id_type, oid) for oid in data["objects"]]

    def get(
        self,
        location: ContainerLocation,
        keys: Collection[ObjectID],
    ) -> Mapping[ObjectID, bytes]:
        body = {
            "savepoint_id": location.savepoint_id,
            "container_id": location.container_id,
            "configuration_id": location.configuration_id,
            "objects": [_object_id_to_str(obj) for obj in keys],
        }
        response = self._session.post(self._join_url("/savepoint/get"), json=body, stream=True)

        obj_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        result: dict[ObjectID, bytes] = {}
        with tarfile.open(fileobj=BufferedReader(response.raw), mode="r") as tar:
            for member, file in tar_iterate_on_members(tar):
                result[_object_id_from_str(obj_id_type, member.name)] = file.read()

        return result

    def add_objects(
        self,
        dependencies: list[PipeDependencies],
        objects: Mapping[ContainerLocation, Mapping[ObjectID, Buffer]],
    ) -> None:
        # Build the dependencies JSON
        deps_json = []
        for dependency in dependencies:
            dependency_entry = {
                "savepoint_range": {
                    "start": dependency.savepoints_range.start,
                    "end": dependency.savepoints_range.end,
                },
                "configuration_id": dependency.configuration,
                "dependencies": [
                    [container_id, _object_id_to_str(object_id), model_path]
                    for container_id, object_id, model_path in dependency.dependencies
                ],
            }
            deps_json.append(dependency_entry)

        # Build the custom_invalidation tar
        custom_invalidation_tar = SpooledTarWriter()
        for dependency in dependencies:
            for container_index, container_data in enumerate(dependency.custom_invalidation):
                for object_id, data in container_data:
                    serialized = _object_id_to_str(object_id)
                    path = (
                        f"{dependency.pipe_id}/{dependency.configuration}/"
                        + f"{container_index}/{serialized}"
                    )
                    custom_invalidation_tar.add_file(path, data)

        # Build the objects tar
        objects_tar = SpooledTarWriter()
        for location, obj_map in objects.items():
            for object_id, content in obj_map.items():
                serialized = _object_id_to_str(object_id)
                path = (
                    f"{location.savepoint_id}/{location.container_id}/"
                    + f"{location.configuration_id}/{serialized}"
                )
                uname, gname = _object_id_serialize_split(object_id)
                objects_tar.add_file(path, content, uname, gname)

        files: _FilesType = {
            "dependencies": ("dependencies", json.dumps(deps_json)),
            "custom_invalidation": ("custom_invalidation", custom_invalidation_tar.get_file()),
            "objects": ("objects", objects_tar.get_file()),
        }
        self._session.post(self._join_url("/savepoint/add-objects"), files=files)

    def get_epoch(self) -> int:
        response = self._session.get(self._join_url("/model/epoch"))
        return response.json()["epoch"]

    def get_model(self) -> tuple[Model, int]:
        response = self._session.get(self._join_url("/model"))
        data = response.json()

        model_type: type[Model] = get_singleton(Model)  # type: ignore[type-abstract]
        model, _ = model_type.deserialize(data["model"].encode())
        return (model, data["epoch"])

    def set_model(
        self,
        new_model: Model,
        changed_paths: ModelPathSet,
        custom_invalidations: list[ObjectsToInvalidate],
        model_bytes: bytes | None = None,
    ) -> SetModelResult:
        invalidation_json = {
            "invalidation_list": list(changed_paths),
            "additional_objects": [
                {
                    "savepoint_range": {
                        "start": obj.savepoint_range.start,
                        "end": obj.savepoint_range.end,
                    },
                    "container_id": obj.container_id,
                    "configuration_id": obj.configuration_id,
                    "objects": [_object_id_to_str(obj_id) for obj_id in obj.objects],
                }
                for obj in custom_invalidations
            ],
        }

        if model_bytes is None:
            model_bytes = new_model.serialize()
        files: _FilesType = {
            "invalidation": ("invalidation", json.dumps(invalidation_json)),
            "model": ("model", model_bytes),
        }
        response = self._session.post(self._join_url("/model/set"), files=files)

        data = response.json()
        epoch = data["epoch"]
        obj_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]
        invalidated: dict[ContainerLocation, set[ObjectID]] = {}
        for entry in data["invalidated"]:
            location = ContainerLocation(
                savepoint_id=entry["savepoint_id"],
                container_id=entry["container_id"],
                configuration_id=entry["configuration"],
            )
            invalidated[location] = {obj_id_type.deserialize(s) for s in entry["object_ids"]}

        return SetModelResult(epoch, invalidated)

    def metadata(self) -> ProjectMetadata:
        response = self._session.get(self._join_url("/metadata"))
        data = response.json()
        return ProjectMetadata(
            version=data["version"],
            pipeline_description_hash=data["pipeline_description_hash"],
            last_fetch=datetime.fromtimestamp(data["last_fetch"]),
            last_object_save=datetime.fromtimestamp(data["last_object_save"]),
            last_model_save=datetime.fromtimestamp(data["last_model_save"]),
        )

    def prune_objects(self):
        raise NotImplementedError

    def put_files_in_storage(self, files: list[FileStorageEntry]) -> list[str]:
        file_tar = SpooledTarWriter()
        for entry in files:
            if entry.contents is not None:
                file_tar.add_file(entry.name, io.BytesIO(entry.contents))
            elif entry.path is not None:
                with open(entry.path, "rb") as f:
                    file_tar.add_file(entry.name, f)
            else:
                raise ValueError

        response = self._session.post(self._join_url("/hashmap/put-file"), data=file_tar.get_file())

        data = response.json()
        name_to_hash: dict[str, str] = {item["name"]: item["hash"] for item in data}
        return [name_to_hash[entry.name] for entry in files]

    def get_files_from_storage(self, requests: list[FileRequest]) -> dict[str, bytes]:
        hashes = [r.hash for r in requests]
        response = self._session.post(self._join_url("/hashmap/get-file"), json=hashes, stream=True)

        result: dict[str, bytes] = {}
        with tarfile.open(fileobj=BufferedReader(response.raw), mode="r") as tar:
            for member, file in tar_iterate_on_members(tar):
                result[member.name] = file.read()

        return result

    def get_custom_invalidation_data(
        self, pipe_id: int, configuration_hash: str
    ) -> PipeCustomInvalidation:
        params: dict[str, int | str] = {
            "pipe_id": pipe_id,
            "configuration_hash": configuration_hash,
        }
        response = self._session.get(
            self._join_url("/savepoint/get-custom-invalidation-data"), params=params, stream=True
        )

        obj_id_type: type[ObjectID] = get_singleton(ObjectID)  # type: ignore[type-abstract]

        result_dict: defaultdict[int, list[tuple[ObjectID, bytes]]] = defaultdict(list)
        with tarfile.open(fileobj=BufferedReader(response.raw), mode="r") as tar:
            for member, file in tar_iterate_on_members(tar):
                # The path format is `${container_index}/${object_id}`
                index_string, object_id_string = member.name.split("/", 1)
                object_id = _object_id_from_str(obj_id_type, object_id_string)
                index = int(index_string)
                result_dict[index].append((object_id, file.read()))

        if len(result_dict) == 0:
            return []

        max_index = max(result_dict.keys())
        return [result_dict.get(i, []) for i in range(max_index + 1)]
