#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import json
import os
import queue
from tempfile import TemporaryDirectory

import revng
from revng.pypeline.daemon.daemon import Daemon
from revng.pypeline.pipeline_parser import load_pipeline_yaml

from .base import Response, TestServer


class WebsocketMock:
    def __init__(self):
        self.queue = queue.Queue()

    def send(self, value: str):
        self.queue.put(value)

    def recv(self) -> str:
        return self.queue.get()


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
            self.pipeline_yaml = pipeline_file.read()
        self.pipeline = load_pipeline_yaml(self.pipeline_yaml)

        self.daemon = Daemon(
            version=revng.__version__,
            pipeline_yaml=self.pipeline_yaml,
            pipeline=self.pipeline,
            debug=True,
            storage_provider_url=storage_provider_url,
            cache_dir=self.cache_dir_tmp.name,
        )

        self.websocket = WebsocketMock()

    def get_epoch(self) -> Response:
        response = asyncio.run(
            self.daemon.get_epoch(
                {
                    "project_id": self.project_id,
                }
            )
        )
        return Response(
            code=response.code,
            body=response.body,
        )

    def get_pipeline(self) -> Response:
        response = self.daemon.get_pipeline()
        return Response(
            code=response.code,
            body=response.body,
        )

    def get_model(self) -> Response:
        response = asyncio.run(
            self.daemon.get_model(
                {
                    "project_id": self.project_id,
                }
            )
        )
        return Response(
            code=response.code,
            body=response.body,
        )

    def run_analysis(self, analysis_request) -> Response:
        analysis_request.setdefault("project_id", self.project_id)
        response = asyncio.run(self.daemon.analyze(analysis_request))
        for notification in response.notifications:
            self.websocket.send(json.dumps(notification))
        return Response(
            code=response.code,
            body=response.body,
        )

    def get_artifact(self, artifact_request) -> Response:
        artifact_request.setdefault("project_id", self.project_id)
        response = asyncio.run(self.daemon.artifact(artifact_request))
        return Response(
            code=response.code,
            body=response.body,
        )

    def subscribe(self):
        return self.websocket
