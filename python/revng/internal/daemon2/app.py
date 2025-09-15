#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import base64
import logging
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.exceptions import HTTPException
from starlette.websockets import WebSocket

from revng.pypeline.pipeline import Pipeline
from revng.pypeline.container import Container
from revng.pypeline.object import ObjectID, Kind
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.task.requests import Requests
from revng.pypeline.storage import storage_provider_factory

from revng.pypeline.utils.registry import get_registry, get_singleton

from .lock import LockManager
from .postoffice import PostOffice
from .utils import compute_objects, compute_artifact


# Global instances
logger = logging.getLogger(__name__)
post_office = PostOffice()


class BasicHTTPException(HTTPException):
    """Custom HTTP exception that includes JSON data"""
    def __init__(self, status_code: int, data: Any):
        super().__init__(status_code=status_code)
        self.data = data


async def basic_exception_handler(request: Request, exc: BasicHTTPException) -> JSONResponse:
    """Handle BasicHTTPException and return JSON response"""
    return JSONResponse(
        content=exc.data,
        status_code=exc.status_code
    )


def get_project_id(headers: dict[str, str]) -> str:
    """Extract project ID from headers, raise exception if missing"""
    if "x-projectid" not in headers:
        raise BasicHTTPException(400, {
            "msg": "X-ProjectId header is required"
        })
    return headers["x-projectid"]


async def invalidation_websocket(websocket: WebSocket):
    """Handle WebSocket connections for invalidations"""
    await websocket.accept()
    subscriber = None

    try:
        project_id = get_project_id(dict(websocket.headers))
        subscriber = await post_office.subscribe(project_id, websocket)
        await subscriber.listen_for_messages()
    except BasicHTTPException as e:
        await websocket.close(code=4000, reason=json.dumps(e.data))
    except Exception as e:
        logger.exception("Uncaught exception ")
        await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
    finally:
        # Clean up the subscription
        if subscriber:
            await post_office.unsubscribe(subscriber)


async def epoch_endpoint(request: Request) -> JSONResponse:
    """Get epoch information for a project"""
    return JSONResponse({
        "epoch": request.app.state.lock.get_epoch(get_project_id(dict(request.headers))),
        "version": request.app.state.version,
    })


async def pipeline_endpoint(request: Request) -> JSONResponse:
    """Get pipeline information"""
    return JSONResponse({
        "epoch": request.app.state.lock.get_epoch(get_project_id(dict(request.headers))),
        "version": request.app.state.version,
        "pipeline": request.app.state.pipeline_raw,
        "containers": [
            {
                "name": name,
                "kind": container.kind.serialize(),
                "is_text": container.is_text(),
            }
            for name, container in get_registry(Container).items()
        ],
        "kinds": [
            {
                "name": kind.serialize(),
                "parent": kind.parent().serialize() if kind.parent() else None,
            }
            for kind in get_singleton(Kind).kinds()
        ],
    })


async def model_endpoint(request: Request) -> JSONResponse:
    """Get model data for a project"""
    project_id = get_project_id(dict(request.headers))
    epoch = request.app.state.lock.get_epoch(project_id)

    async with LockManager(
        lock=request.app.state.lock,
        project_id=project_id,
        epoch=epoch,
        metadata={"task": "get_model"},
        increase_epoch=False,  # We don't want to increase the epoch as we only read the model
    ):
        # Create a storage provider
        storage_provider = storage_provider_factory(
            model_path=project_id,
            storage_url=request.app.state.storage_provider,
        )
        model_ty: type[Model] = get_singleton(Model)
        model: bytes = storage_provider.get_model()

        # Optimize the model serialization
        if model_ty.is_text():
            model_str = model.decode("utf-8")
        else:
            model_str = base64.b64encode(model).decode("utf-8")

        return JSONResponse({
            "epoch": epoch,
            "is_text": model_ty.is_text(),
            "model": model_str,
        })


async def artifact_endpoint(request: Request) -> JSONResponse:
    """Process artifact requests"""
    project_id = get_project_id(dict(request.headers))

    # Parse JSON data
    content_type = request.headers.get("content-type", "")
    if content_type != "application/json":
        raise BasicHTTPException(400, {"msg": "Content-Type must be application/json"})

    try:
        data: dict = await request.json()
    except json.JSONDecodeError:
        raise BasicHTTPException(400, {"msg": "Invalid JSON data"})

    model_ty: type[Model] = get_singleton(Model)
    pipeline: Pipeline = request.app.state.pipeline

    # Extract the data
    epoch = data["epoch"]
    artifacts = data["artifacts"]

    # Validate data
    for artifact_name, _ in artifacts.items():
        if artifact_name not in pipeline.artifacts:
            raise BasicHTTPException(400, {
                "msg": f"Artifact {artifact_name} not found in the pipeline.",
                "available_artifacts": list(pipeline.artifacts.keys()),
            })

    # Acquire the lock for the project
    async with LockManager(
        lock=request.app.state.lock,
        project_id=project_id,
        epoch=epoch,
        metadata={"artifacts": artifacts},
        increase_epoch=False,  # It doesn't modify the model
    ):
        # Create a storage provider
        storage_provider = storage_provider_factory(
            model_path=project_id,
            storage_url=request.app.state.storage_provider,
        )
        # Load the model
        model = model_ty()
        model.deserialize(storage_provider.get_model())

        # Process each artifact
        res = {}
        for artifact_name, artifact_data in artifacts.items():
            # Compute the artifact
            res[artifact_name] = compute_artifact(
                storage_provider=storage_provider,
                pipeline=pipeline,
                model=ReadOnlyModel(model),
                artifact_name=artifact_name,
                artifact_data=artifact_data,
            )

        # Return the artifacts
        return JSONResponse({"artifacts": res})



async def monitoring_endpoint(request: Request) -> JSONResponse:
    """Get monitoring information about WebSocket subscribers"""
    total_subscribers = await post_office.get_total_subscribers()

    # Get per-project subscriber counts
    project_counts = {}
    async with post_office._lock:
        for project_id, subscribers in post_office.subscribers.items():
            active_count = sum(1 for s in subscribers if s.is_active)
            if active_count > 0:
                project_counts[project_id] = active_count

    return JSONResponse({
        "total_subscribers": total_subscribers,
        "project_subscribers": project_counts,
        "active_projects": len(project_counts)
    })


async def analysis_endpoint(request: Request) -> JSONResponse:
    """Process analysis requests"""
    project_id = get_project_id(dict(request.headers))

    # Parse JSON data
    content_type = request.headers.get("content-type", "")
    if content_type != "application/json":
        raise BasicHTTPException(400, {"msg": "Content-Type must be application/json"})

    try:
        data: dict = await request.json()
    except json.JSONDecodeError:
        raise BasicHTTPException(400, {"msg": "Invalid JSON data"})

    model_ty: type[Model] = get_singleton(Model)
    pipeline: Pipeline = request.app.state.pipeline

    # Extract the data
    epoch = data["epoch"]
    analysis = data["analysis"]
    configuration = data.get("configuration", "")
    pipeline_configuration = data.get("pipeline_configuration", {})
    containers = data.get("containers", {})

    # Validate data
    if analysis not in pipeline.analyses:
        raise BasicHTTPException(400, {
            "msg": f"Analysis {analysis} not found in the pipeline.",
            "available_analyses": list(pipeline.analyses.keys()),
        })

    for container_name, objects in containers.items():
        for decl in pipeline.declarations:
            if container_name == decl.name:
                break
        else:
            raise BasicHTTPException(400, {
                "msg": f"Container {container_name} not found in the pipeline.",
                "available_containers": list(sorted(decl.name for decl in pipeline.declarations)),
            })

    # Acquire the lock for the project
    lock_manager = None
    async with LockManager(
        lock=request.app.state.lock,
        project_id=project_id,
        epoch=epoch,
        metadata={
            "analysis": analysis,
            "configuration": configuration,
            "pipeline_configuration": pipeline_configuration,
            "containers": containers,
        },
    ) as lock_manager:
        # Create a storage provider
        storage_provider = storage_provider_factory(
            model_path=project_id,
            storage_url=request.app.state.storage_provider,
        )
        # Load the model
        model = model_ty()
        model.deserialize(storage_provider.get_model())

        # Setup the requests
        requests = Requests()
        for binding in pipeline.analyses[analysis].bindings:
            kind: Kind = binding.container_type.kind
            objects = containers.get(binding.name)
            if objects is not None and not isinstance(objects, list):
                raise BasicHTTPException(400, {
                    "msg": f"Objects for container {binding.name} must be a list, got {type(objects)}",
                })
            requests.insert(binding, compute_objects(model, kind, objects))

        # Run the analysis
        new_model = pipeline.run_analysis(
            model=ReadOnlyModel(model),
            analysis_name=analysis,
            requests=requests,
            analysis_configuration=configuration,
            pipeline_configuration=pipeline_configuration,
            storage_provider=storage_provider,
        )
        # Save the new model
        new_model_bytes = storage_provider.get_model()

    if model_ty.is_text():
        new_model_str = new_model_bytes.decode("utf-8")
    else:
        new_model_str = base64.b64encode(new_model_bytes).decode("utf-8")

    diff = str(model.diff(new_model))
    # Notify subscribers
    await post_office.notify(project_id, json.dumps({
        "type": "analysis",
        "analysis": analysis,
        "epoch": lock_manager.epoch,
        "new_model": new_model_str,
        "invalidated": diff,
    }))

    # Return the updated model
    return JSONResponse({
        "epoch": lock_manager.epoch,
        "model-diff": diff,
    })



# Define routes
routes = [
    Route("/api/epoch", epoch_endpoint, methods=["GET"]),
    Route("/api/pipeline", pipeline_endpoint, methods=["GET"]),
    Route("/api/model", model_endpoint, methods=["GET"]),
    Route("/api/artifact", artifact_endpoint, methods=["POST"]),
    Route("/api/analysis", analysis_endpoint, methods=["POST"]),
    Route("/api/monitoring", monitoring_endpoint, methods=["GET"]),
    WebSocketRoute("/api/subscribe", invalidation_websocket),
]

# Create the Starlette application
app = Starlette(
    debug=False,
    routes=routes,
    exception_handlers={
        BasicHTTPException: basic_exception_handler,
    }
)
