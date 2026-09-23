#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import socket
import time
from contextlib import AbstractContextManager
from multiprocessing import Process
from signal import SIGTERM
from tempfile import TemporaryDirectory

import httpx2
from httpx2.websockets import WebSocketSession

from revng.pypeline.main import main

from .base import Response, TestServer

logger = logging.getLogger(__name__)


class WebSocketAdapter:
    """
    Adapts an httpx2 WebSocket session to `SubscribeConnection`. The session is
    only available as a context manager, which is entered here and left on
    `close`.
    """

    def __init__(self, context_manager: AbstractContextManager[WebSocketSession]):
        self.context_manager = context_manager
        self.session = context_manager.__enter__()

    def recv(self) -> bytes:
        return self.session.receive_bytes()

    def close(self):
        self.context_manager.__exit__(None, None, None)


def find_free_port():
    """Find a free port to use for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class StarletteTestServer(TestServer):
    """Helper class to start and stop the daemon for testing."""

    def __init__(self, port=None, storage_provider_url: str = "memory://"):
        super().__init__()
        self.port = port or find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.storage_provider_url = storage_provider_url
        # Create temporary files for the DB and the config that points to the DB
        self.cache_dir = TemporaryDirectory()
        logger.info("Working with cache directory at %s", self.cache_dir)

        # The server daemon has to be a daemon so when the tests finish it will
        # be killed. But this silences the exceptions, so if it fails, you need
        # to run it manually to fix them.
        self.server_process: Process | None = Process(target=self._run_server, daemon=True)
        # `start` pickles `self` to hand it over to the child process, so the
        # client, which is not picklable, can only be built afterwards.
        self.server_process.start()

        # Configure a client that can directly talk to the daemon
        self.session = httpx2.Client(http2=True, timeout=None)
        self._wait_for_server()

    def __del__(self):
        self.stop()

    def stop(self):
        if self.server_process is not None:
            self.server_process.terminate()
            self.server_process.join()
            assert self.server_process.exitcode in (0, -SIGTERM)
            self.server_process = None

    def _run_server(self):
        """
        This runs in a separate process that calls the cli to spawn the daemon.
        Beware that if it raises an exception it will not be print on stdout or
        stderr, so if the daemon doesn't start, try to run it manually.
        """
        logger.info("Starting the daemon on port %s", self.port)
        main(
            (
                "-C",
                self.tmp_dir.name,
                "--pipebox",
                self.pipebox_path,
                "project",
                "--storage-provider",
                self.storage_provider_url,
                "--pipeline",
                self.pipeline_path,
                "--cache-dir",
                self.cache_dir.name,
                "daemon",
                "--bind",
                f"127.0.0.1:{self.port!s}",
            )
        )
        logger.critical("Server exiting")

    def _wait_for_server(self, timeout=10):
        """Wait for the server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            logger.info("Waiting for the daemon to startup... ")
            try:
                response = self.session.get(f"{self.base_url}/status", timeout=1)
                if response.status_code == 200:
                    return
            except httpx2.HTTPError:
                pass
            time.sleep(1)
        raise RuntimeError(f"Server did not start within {timeout} seconds")

    def get_epoch(self) -> Response:
        logger.info("Getting epoch")
        r = self.session.get(f"{self.base_url}/api/epoch")
        logger.info("Epoch response: %s", r.text)
        return Response(code=r.status_code, body=r.json())

    def get_pipeline(self) -> Response:
        logger.info("Getting pipeline")
        r = self.session.get(f"{self.base_url}/api/pipeline")
        logger.info("Pipeline response: %s", r.text)
        return Response(code=r.status_code, body=r.json())

    def get_model(self) -> Response:
        logger.info("Getting model")
        r = self.session.get(f"{self.base_url}/api/model")
        logger.info("Model response: %s", r.text)
        return Response(code=r.status_code, body=r.json())

    def put_file(self, put_file_request) -> Response:
        logger.info("Putting file in storage with request %s", put_file_request)
        r = self.session.post(
            f"{self.base_url}/api/put-file",
            files={"file": (put_file_request["name"], put_file_request["contents"])},
        )
        logger.info("put_file response: %s", r.text)
        return Response(code=r.status_code, body=r.json())

    def run_analysis(self, analysis_request) -> Response:
        logger.info("Running analysis with request %s", analysis_request)
        r = self.session.post(f"{self.base_url}/api/analysis", json=analysis_request)
        logger.info("Analysis response: %s", r.text)
        return Response(code=r.status_code, body=r.json())

    def get_artifact(self, artifact_request) -> Response:
        logger.info("Getting Artifact with request %s", artifact_request)
        r = self.session.post(f"{self.base_url}/api/artifact", json=artifact_request)
        logger.info("Artifact response: %s", r.text)
        return Response(code=r.status_code, body=r.json())

    def subscribe(self):
        return WebSocketAdapter(
            self.session.websocket(f"ws://127.0.0.1:{self.port}/api/notifications")
        )
