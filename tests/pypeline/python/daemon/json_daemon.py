#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import functools
import os
from tempfile import TemporaryDirectory

from revng.pypeline.daemon.daemon import Daemon
from revng.pypeline.daemon.daemon import Response as DaemonResponse
from revng.pypeline.daemon.exceptions import DaemonException
from revng.pypeline.pipeline_parser import load_pipeline_yaml
from revng.pypeline.storage.notification_queue import LOCAL_QUEUE
from revng.pypeline.utils import PypelineException

from .base import Response, TestServer


class WebsocketMock:
    def __init__(self):
        self.queue = LOCAL_QUEUE.get_queue()

    def recv(self) -> bytes:
        return asyncio.run(self.queue.get())


def handle_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DaemonException as e:
            return DaemonResponse(e.code, e.body)
        except PypelineException as e:
            return DaemonResponse(500, {"message": str(e)})

    return wrapper


class JsonTestServer(TestServer):
    """A test server that talks through JSON and doesn't involve
    neither sockets nor HTTP."""

    def __init__(self, storage_provider_url: str = "memory://"):
        super().__init__()

        # We have to change the path so the local provider can find
        # the model
        os.chdir(self.tmp_dir_path)

        self.cache_dir_tmp = TemporaryDirectory()
        self.project_id = "test_project_id_json"

        with open(self.pipeline_path, "r") as pipeline_file:
            self.pipeline = load_pipeline_yaml(pipeline_file.read())

        self.daemon = Daemon(
            pipeline=self.pipeline,
            storage_provider_url=storage_provider_url,
            cache_dir=self.cache_dir_tmp.name,
            base_directory=self.tmp_dir_path,
        )

        self.websocket = WebsocketMock()

    @handle_exceptions
    def get_epoch(self) -> Response:
        response = asyncio.run(self.daemon.get_epoch({"project_id": self.project_id}))
        return Response(code=response.code, body=response.body)

    @handle_exceptions
    def get_pipeline(self) -> Response:
        response = self.daemon.get_pipeline()
        return Response(code=response.code, body=response.body)

    @handle_exceptions
    def get_model(self) -> Response:
        response = asyncio.run(self.daemon.get_model({"project_id": self.project_id}))
        return Response(code=response.code, body=response.body)

    @handle_exceptions
    def put_file(self, put_file_request) -> Response:
        put_file_request.setdefault("project_id", self.project_id)
        response = asyncio.run(self.daemon.put_file(put_file_request))
        return Response(code=response.code, body=response.body)

    @handle_exceptions
    def run_analysis(self, analysis_request) -> Response:
        analysis_request.setdefault("project_id", self.project_id)
        response = asyncio.run(self.daemon.analyze(analysis_request))
        return Response(code=response.code, body=response.body)

    @handle_exceptions
    def get_artifact(self, artifact_request) -> Response:
        artifact_request.setdefault("project_id", self.project_id)
        response = asyncio.run(self.daemon.artifact(artifact_request))
        return Response(code=response.code, body=response.body)

    def subscribe(self):
        return self.websocket
