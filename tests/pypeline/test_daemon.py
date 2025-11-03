#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import json
import logging
import sys

# Import the pipebox, even if unused it will populate the registries
import pipebox as _  # noqa: F401
import pytest
from daemon.base import TestServer
from daemon.json_daemon import JsonTestServer
from daemon.starlette_daemon import StarletteTestServer

import revng
from revng.pypeline import initialize_pypeline

initialize_pypeline()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@pytest.fixture(params=["memory://", "local://"])
def storage_provider_url(request):
    return request.param


@pytest.fixture(params=["json", "starlette"])
def daemon_server(request, storage_provider_url):
    """Fixture that provides a running daemon server for testing."""
    if request.param == "json":
        return JsonTestServer(
            storage_provider_url=storage_provider_url,
        )
    elif request.param == "starlette":
        return StarletteTestServer(
            storage_provider_url=storage_provider_url,
        )
    raise ValueError(f"Unknown daemon type {request.param}")


def test_daemon(daemon_server: TestServer):
    # Test epoch endpoint
    logger.info("Testing epoch endpoint")
    response = daemon_server.get_epoch()
    assert response.code == 200
    epoch_data = response.body
    assert "epoch" in epoch_data
    current_epoch = epoch_data["epoch"]

    # Test pipeline endpoint
    logger.info("Testing pipeline endpoint")
    response = daemon_server.get_pipeline()
    pipeline_data = response.body
    assert "version" in pipeline_data
    assert "pipeline" in pipeline_data
    assert "containers" in pipeline_data
    assert "kinds" in pipeline_data
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
    assert response.code == 200
    model_data = response.body
    assert "epoch" in model_data
    assert "is_text" in model_data
    assert "model" in model_data
    assert model_data["is_text"]  # DictModel is text-based
    initial_model = model_data["model"]

    # Connect to the websocket
    notifications_websocket = daemon_server.subscribe()

    # Test analysis endpoint - run init_analysis
    logger.info("Testing analysis endpoint")
    response = daemon_server.run_analysis(
        {
            "epoch": current_epoch,
            "analysis": "init_analysis",
            "configuration": "",
            "pipeline_configuration": {},
            "containers": {
                # Empty list means all objects of this container
                "child_source": []
            },
        }
    )
    assert response.code == 200
    analysis_data = response.body
    assert "epoch" in analysis_data
    assert "diff" in analysis_data
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
    assert response.code == 200
    updated_model_data = response.body
    assert updated_model_data["epoch"] == new_epoch
    updated_model = updated_model_data["model"]
    # Model should have changed
    assert updated_model != initial_model

    # Test artifact endpoint - request ChildArtifact
    logger.info("Testing artifact endpoint")
    response = daemon_server.get_artifact(
        {"epoch": new_epoch, "artifacts": {"ChildArtifact": {}}}  # Empty data for the artifact
    )
    assert response.code == 200
    artifact_data = response.body
    assert "artifacts" in artifact_data
    assert "ChildArtifact" in artifact_data["artifacts"]

    # Test another analysis - blackhole
    logger.info("Testing blackhole")
    response = daemon_server.run_analysis(
        {
            "epoch": new_epoch,
            "analysis": "blackhole",
            "configuration": "",
            "pipeline_configuration": {},
            "containers": {"root_source": []},
        }
    )
    assert response.code == 200
    purge_data = response.body
    assert "epoch" in purge_data
    assert "diff" in purge_data
    final_epoch = purge_data["epoch"]
    assert final_epoch > new_epoch

    # Test error handling - invalid analysis
    logger.info("Testing error handling with invalid analysis")
    response = daemon_server.run_analysis(
        {
            "epoch": final_epoch,
            "analysis": "NonExistentAnalysis",
            "configuration": "",
            "containers": {},
        }
    )
    assert response.code == 400
    error_data = response.body
    assert "msg" in error_data
    assert "available_analyses" in error_data

    # Test error handling - invalid artifact
    logger.info("Testing error handling with invalid artifact")
    response = daemon_server.get_artifact(
        {"epoch": final_epoch, "artifacts": {"NonExistentArtifact": {}}}
    )
    logger.info("NonExistentArtifact response: %s", response.body)
    assert response.code == 400
    error_data = response.body
    assert "msg" in error_data
    assert "available_artifacts" in error_data

    # Verify final state
    logger.info("Verifying final daemon state")
    response = daemon_server.get_epoch()
    assert response.code == 200
    final_epoch_data = response.body
    assert final_epoch_data["epoch"] == final_epoch

    logger.info("All daemon tests completed successfully!")
