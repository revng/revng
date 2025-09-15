#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import sys
import json
import time
import socket
import logging
import threading
from typing import Any
from pathlib import Path

import pytest
import uvicorn
import requests
import websockets.sync.client
from tempfile import NamedTemporaryFile, TemporaryDirectory

import revng
from revng.internal.cli.revng2 import main
from revng.pypeline import initialize_pypeline
from revng.pypeline.pipeline_parser import load_pipeline_yaml_file

import simple_pipeline
initialize_pypeline()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def find_free_port():
    """Find a free port to use for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


DAEMON_CONFIG = """
storage_provider: "sqlite://{storage_provider_path}"

lock:
    class: "revng.internal.daemon2.lock.local_lock:LocalLock"
    args:
        lock_duration: 60
"""

class DaemonTestServer:
    """Helper class to start and stop the daemon for testing."""

    def __init__(self, port=None):
        self.port = port or find_free_port()
        self.server_thread = None
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.project_id = "test_project_id"

        this_dir = Path(__file__).parent
        self.pipebox_path = str(this_dir / "simple_pipeline.py")
        self.pipeline_path = str(this_dir / "pipeline.yml")
        # Create temporary files for the DB and the config that points to the DB
        self.db_dir = TemporaryDirectory()
        logger.info("Working with DB directory at %s", self.db_dir)
        self.config_file = NamedTemporaryFile()
        logger.info("Working with daemon config file at %s", self.config_file.name)
        self.config_file.truncate()
        configs = DAEMON_CONFIG.format(storage_provider_path=Path(self.db_dir.name) / "cache.db")
        logger.info("Working with daemon configs: %s",configs)
        self.config_file.write(configs.encode("utf-8"))
        self.config_file.flush()
        self.config_path = self.config_file.name

        # Configure a session that can directly talk to the daemon
        self.session = requests.Session()
        self.session.headers["X-Projectid"] = self.project_id

        # The server thread has to be a daemon so when the tests finish it will be killed
        # self._run_server() # Decomment to debug why the daemon doesn't start
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        self._wait_for_server()

    def _run_server(self):
        """
        This is the thread that calls the cli to spawn the daemon.
        Beware that if it raises an exception it will not be print on stdout or
        stderr, so if the daemon doesn't start, try to run it manually.
        """
        logger.info("Starting the daemon on port %s", self.port)
        main((
            "--pipebox",
            self.pipebox_path,
            "daemon",
            "--pipeline",
            self.pipeline_path,
            "--port",
            str(self.port),
            "--config",
            self.config_path,
        ))
        logger.critical("Server exiting")

    def _wait_for_server(self, timeout=10):
        """Wait for the server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            logger.info("Waiting for the daemon to startup... ")
            try:
                response = self.session.get(f"{self.base_url}/api/epoch", timeout=1)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError(f"Server did not start within {timeout} seconds")

    def get_epoch(self) -> requests.Response:
        logger.info("Getting epoch")
        response = self.session.get(f"{self.base_url}/api/epoch")
        logger.info("Epoch response: %s", response.text)
        return response

    def get_pipeline(self) -> requests.Response:
        logger.info("Getting pipeline")
        response = self.session.get(f"{self.base_url}/api/pipeline")
        logger.info("Pipeline response: %s", response.text)
        return response

    def get_model(self) -> requests.Response:
        logger.info("Getting model")
        response = self.session.get(f"{self.base_url}/api/model")
        logger.info("Model response: %s", response.text)
        return response

    def get_monitoring(self) -> requests.Response:
        logger.info("Getting model")
        response = self.session.get(f"{self.base_url}/api/monitoring")
        logger.info("Monitoring response: %s", response.text)
        return response


    def run_analysis(self, analysis_request) -> requests.Response:
        logger.info("Running analysis with request %s", analysis_request)
        response = self.session.post(
            f"{self.base_url}/api/analysis",
            json=analysis_request,
        )
        logger.info("Analysis response: %s", response.text)
        return response

    def get_artifact(self, artifact_request) -> requests.Response:
        logger.info("Getting Artifact with request %s", artifact_request)
        response = self.session.post(
            f"{self.base_url}/api/artifact",
            json=artifact_request,
        )
        logger.info("Artifact response: %s", response.text)
        return response

    def subscribe(self) -> websockets.sync.Websocket:
        return websockets.sync.client.connect(
            f'ws://127.0.0.1:{self.port}/api/subscribe',
            additional_headers={"X-ProjectId": self.project_id},
        )


@pytest.fixture
def daemon_server():
    """Fixture that provides a running daemon server for testing."""
    return DaemonTestServer()


def test_daemon(daemon_server):
    # Test epoch endpoint
    logger.info("Testing epoch endpoint")
    response = daemon_server.get_epoch()
    assert response.status_code == 200
    epoch_data = response.json()
    assert "epoch" in epoch_data
    assert "version" in epoch_data
    assert epoch_data["version"] == revng.__version__
    current_epoch = epoch_data["epoch"]


    # Test pipeline endpoint
    logger.info("Testing pipeline endpoint")
    response = daemon_server.get_pipeline()
    pipeline_data = response.json()
    assert "epoch" in pipeline_data
    assert "version" in pipeline_data
    assert "pipeline" in pipeline_data
    assert "containers" in pipeline_data
    assert "kinds" in pipeline_data
    assert pipeline_data["epoch"] == current_epoch
    assert pipeline_data["version"] == revng.__version__


    # Validate containers structure
    containers = pipeline_data["containers"]
    assert isinstance(containers, list)
    container_names = [c["name"] for c in containers]
    expected_containers = ["RootDictContainer", "ChildDictContainer"]
    for expected in expected_containers:
        assert expected in container_names, f"Expected container {expected} not found"


    # Validate kinds structure
    kinds = pipeline_data["kinds"]
    assert isinstance(kinds, list)
    kind_names = [k["name"] for k in kinds]
    expected_kinds = ["ROOT", "CHILD", "GRANDCHILD", "CHILD2"]
    for expected in expected_kinds:
        assert expected in kind_names, f"Expected kind {expected} not found"


    # Test model endpoint
    logger.info("Testing model endpoint")
    response = daemon_server.get_model()
    assert response.status_code == 200
    model_data = response.json()
    assert "epoch" in model_data
    assert "is_text" in model_data
    assert "model" in model_data
    assert model_data["is_text"] == True  # DictModel is text-based
    initial_model = model_data["model"]


    # Test monitoring endpoint
    logger.info("Testing monitoring endpoint")
    response = daemon_server.get_monitoring()
    assert response.status_code == 200
    monitoring_data = response.json()
    assert monitoring_data["total_subscribers"] == 0
    assert monitoring_data["project_subscribers"] == {}
    assert monitoring_data["active_projects"] == 0


    # Connect to the websocket
    notifications_websocket = daemon_server.subscribe()

    # Test monitoring endpoint after subscription
    logger.info("Testing monitoring endpoint")
    response = daemon_server.get_monitoring()
    assert response.status_code == 200
    logger.info("Monitoring response: %s", response.text)
    monitoring_data = response.json()
    assert monitoring_data["total_subscribers"] == 1
    assert monitoring_data["project_subscribers"] == {
        daemon_server.project_id: 1,
    }
    assert monitoring_data["active_projects"] == 1


    # Test analysis endpoint - run init_analysis
    logger.info("Testing analysis endpoint")
    response = daemon_server.run_analysis({
        "epoch": current_epoch,
        "analysis": "init_analysis",
        "configuration": "",
        "pipeline_configuration": {},
        "containers": {
            # Empty list means all objects of this container
            "child_source": []
        }
    })
    assert response.status_code == 200
    analysis_data = response.json()
    assert "epoch" in analysis_data
    assert "model-diff" in analysis_data
    new_epoch = analysis_data["epoch"]
    # Epoch should increase after modification
    assert new_epoch > current_epoch


    # Check the analysis notification
    analysis_notification_text = notifications_websocket.recv()
    logger.info("Analysis notification: %s", analysis_notification_text)
    analysis_notification = json.loads(analysis_notification_text)
    assert analysis_notification["type"] == "analysis"
    assert analysis_notification["analysis"] == "init_analysis"
    assert analysis_notification["epoch"] == new_epoch


    # Verify model was modified by getting it again
    logger.info("Verifying model was modified")
    response = daemon_server.get_model()
    assert response.status_code == 200
    updated_model_data = response.json()
    assert updated_model_data["epoch"] == new_epoch
    updated_model = updated_model_data["model"]
    # Model should have changed
    assert updated_model != initial_model
    assert analysis_notification["new_model"] == updated_model


    # Test artifact endpoint - request ChildArtifact
    logger.info("Testing artifact endpoint")
    response = daemon_server.get_artifact({
        "epoch": new_epoch,
        "artifacts": {
            "ChildArtifact": {}  # Empty data for the artifact
        }
    })
    assert response.status_code == 200
    artifact_data = response.json()
    assert "artifacts" in artifact_data
    assert "ChildArtifact" in artifact_data["artifacts"]


    # Test another analysis - blackhole
    logger.info("Testing blackhole")
    response = daemon_server.run_analysis({
        "epoch": new_epoch,
        "analysis": "blackhole",
        "configuration": "",
        "pipeline_configuration": {},
        "containers": {
            "root_source": []
        }
    })
    assert response.status_code == 200
    purge_data = response.json()
    assert "epoch" in purge_data
    assert "model-diff" in purge_data
    final_epoch = purge_data["epoch"]
    assert final_epoch > new_epoch


    # Test error handling - invalid analysis
    logger.info("Testing error handling with invalid analysis")
    response = daemon_server.run_analysis({
        "epoch": final_epoch,
        "analysis": "NonExistentAnalysis",
        "configuration": "",
        "containers": {}
    })
    assert response.status_code == 400
    error_data = response.json()
    assert "msg" in error_data
    assert "available_analyses" in error_data

    # Test error handling - invalid artifact
    logger.info("Testing error handling with invalid artifact")
    response = daemon_server.get_artifact({
        "epoch": final_epoch,
        "artifacts": {
            "NonExistentArtifact": {}
        }
    })
    logger.info("NonExistentArtifact response: %s", response.text)
    assert response.status_code == 400
    error_data = response.json()
    assert "msg" in error_data
    assert "available_artifacts" in error_data

    # Test error handling - missing project ID header
    logger.info("Testing error handling with missing project ID")
    session_no_project = requests.Session()
    response = session_no_project.get(f"{daemon_server.base_url}/api/epoch")
    logger.info(" missing project ID response: %s", response.text)
    assert response.status_code == 400
    error_data = response.json()
    assert "msg" in error_data
    assert "X-ProjectId" in error_data["msg"]

    # Test error handling - invalid content type for POST endpoints
    logger.info("Testing error handling with invalid content type")
    response = daemon_server.session.post(
        f"{daemon_server.base_url}/api/analysis",
        data="not json",
        headers={"Content-Type": "text/plain"}
    )
    logger.info("invalid content type response: %s", response.text)
    assert response.status_code == 400
    error_data = response.json()
    assert "msg" in error_data
    assert "application/json" in error_data["msg"]

    # Verify final state
    logger.info("Verifying final daemon state")
    response = daemon_server.get_epoch()
    assert response.status_code == 200
    final_epoch_data = response.json()
    assert final_epoch_data["epoch"] == final_epoch

    logger.info("All daemon tests completed successfully!")
