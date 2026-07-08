#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import ParseResult, urlparse

from requests import RequestException, Response, Session

from revng.pypeline.container import Container, ContainerFormat
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.pipeline import AnalysisList, Artifact, Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.storage_provider import FileStorageEntry, InvalidatedObjects, LockType
from revng.pypeline.storage.storage_provider import ProjectID
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils import PypelineException, string_to_bytes
from revng.pypeline.utils.registry import get_singleton

from .backend import Backend, BackendFactory

# `daemon://` is our selector; the wire protocol underneath is plain HTTP.
DAEMON_SCHEME = "daemon"


def _serialize_configuration(configuration: PipelineConfiguration) -> dict[str, str]:
    """
    Turn the CLI's `{Pipe | Analysis: str}` configuration into the `{name: str}`
    shape the daemon expects (the inverse of `deserialize_configuration`).
    """
    return {task.name: value for task, value in configuration.items()}


class DaemonBackend(Backend):
    """A backend backed by the daemon's HTTP API."""

    def __init__(self, base_url: ParseResult, http: Session, pipeline: Pipeline):
        self._base_url = base_url
        self._http = http
        self._pipeline = pipeline
        # The epoch the daemon was at when we last synced the model. Passed back
        # on artifact/analysis requests so the daemon rejects our request (with a
        # 409) if the model changed under us instead of computing inconsistently.
        self._epoch: int | None = None

    def get_model(self, configuration: PipelineConfiguration) -> tuple[Model, int]:
        data = self._get("/api/model").json()
        model_type = get_singleton(Model)  # type: ignore[type-abstract]
        model_bytes = string_to_bytes(data["model"], model_type.is_text())
        model = model_type.deserialize(model_bytes)[0]
        self._epoch = data["epoch"]
        return model, data["epoch"]

    def get_artifact(
        self,
        model: ReadOnlyModel,
        artifact: Artifact,
        requests: ObjectSet,
        configuration: PipelineConfiguration,
    ) -> Container:
        # The regular CLI produces exactly the resolved object set; an empty set
        # means "produce nothing". The daemon reads an empty object list as
        # "all", so short-circuit here to stay coherent with local behavior.
        if len(requests) == 0:
            return artifact.container.container_type()

        body: dict[str, Any] = {
            "artifact": artifact.name,
            "epoch": self._current_epoch(),
            "objects": [object_id.serialize() for object_id in requests],
            "format": "tar",
        }
        serialized_configuration = _serialize_configuration(configuration)
        if serialized_configuration:
            body["configuration"] = serialized_configuration

        response = self._post("/api/artifact", json=body)
        container_type = artifact.container.container_type
        return container_type.from_bytes(response.content, ContainerFormat.TAR)

    def run_analysis(
        self,
        model: ReadOnlyModel,
        analysis_name: str,
        requests: Requests,
        configuration: PipelineConfiguration,
    ) -> tuple[Model, InvalidatedObjects]:
        containers = {
            declaration.name: [object_id.serialize() for object_id in object_set]
            for declaration, object_set in requests.items()
        }
        return self._analyze(analysis_name, configuration, containers)

    def run_analysis_list(
        self,
        model: ReadOnlyModel,
        analysis_list: AnalysisList,
        configuration: PipelineConfiguration,
    ) -> tuple[Model, InvalidatedObjects]:
        # The daemon expands the list itself, running every analysis on all
        # objects, so no per-container object list is needed.
        return self._analyze(analysis_list.name, configuration, {})

    def put_files(self, files: list[FileStorageEntry]) -> list[str]:
        hashes = []
        for entry in files:
            if entry.path is not None:
                with open(entry.path, "rb") as file:
                    response = self._post("/api/put-file", files={"file": (entry.name, file)})
            else:
                response = self._post("/api/put-file", files={"file": (entry.name, entry.contents)})
            hashes.append(response.json()["hash"])
        return hashes

    def _analyze(
        self,
        analysis: str,
        configuration: PipelineConfiguration,
        containers: dict[str, list[str]],
    ) -> tuple[Model, InvalidatedObjects]:
        body = {
            "analysis": analysis,
            "epoch": self._current_epoch(),
            "configuration": _serialize_configuration(configuration),
            "containers": containers,
        }
        response = self._post("/api/analysis", json=body).json()
        self._epoch = response["epoch"]
        # The daemon returns only a diff, so re-fetch the full model. It does not
        # report invalidated objects (the CLI rejects --invalidations against a
        # daemon, since the backend lacks BackendFeature.INVALIDATIONS), so return
        # an empty set.
        new_model, _ = self.get_model(configuration)
        return new_model, InvalidatedObjects()

    def _current_epoch(self) -> int:
        if self._epoch is None:
            self._epoch = self._get("/api/epoch").json()["epoch"]
        return self._epoch

    def _get(self, path: str, **kwargs) -> Response:
        return self._request("GET", path, **kwargs)

    def _post(self, path: str, **kwargs) -> Response:
        return self._request("POST", path, **kwargs)

    def _request(self, method: str, path: str, **kwargs) -> Response:
        try:
            return self._http.request(method, self._url(path), **kwargs)
        except RequestException as exception:
            # A transport-level failure (e.g. the daemon is not running); HTTP
            # error statuses are handled by the response hook instead.
            raise PypelineException(
                f"Could not reach the daemon at {self._base_url.geturl()}: {exception}"
            ) from exception

    def _url(self, path: str) -> str:
        new_path = os.path.normpath(self._base_url.path + path)
        new_path = re.sub(r"^/+", "/", new_path)
        return self._base_url._replace(path=new_path).geturl()


class DaemonBackendFactory(BackendFactory):
    """Delegates all compute to a running daemon at a `daemon://host:port` URL."""

    # Compute runs on the daemon: there is nothing local to inspect (no --debug),
    # the daemon manages its own project (no init) and only returns a model diff
    # (no --invalidations). It therefore provides none of the optional features.

    @classmethod
    def schemes(cls) -> list[str]:
        return [DAEMON_SCHEME]

    def __init__(self, url: str, *, pipeline: Pipeline, base_directory: Path, cache_dir: str):
        # `base_directory` and `cache_dir` are part of the common engine
        # constructor but are meaningless for a remote daemon, which owns its own
        # storage; they are accepted and ignored.
        parsed = urlparse(url)
        if parsed.scheme == DAEMON_SCHEME:
            parsed = parsed._replace(scheme="http")
        self._base_url = parsed
        self._pipeline = pipeline

    @asynccontextmanager
    async def session(
        self,
        *,
        lock_type: LockType,
        project_id: ProjectID | None,
        token: str | None,
        runner_context: RunnerContext = RunnerContext(),
    ) -> AsyncIterator[Backend]:
        http = Session()
        if project_id is not None:
            http.headers["x-project-id"] = project_id
        if token is not None:
            http.headers["authorization"] = f"Bearer {token}"
        http.hooks["response"].append(self._raise_for_status)

        try:
            yield DaemonBackend(self._base_url, http, self._pipeline)
        finally:
            http.close()

    @staticmethod
    def _raise_for_status(response: Response, *args, **kwargs) -> None:
        if response.status_code < 400:
            return
        message = f"The daemon returned an error (HTTP {response.status_code})"
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and "message" in body:
            message = f"{message}: {body['message']}"
        raise PypelineException(message)
