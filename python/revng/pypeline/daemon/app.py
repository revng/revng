#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import logging
import os
import traceback
from functools import wraps
from typing import Any

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from .daemon import Daemon, Response
from .notification_broker import WebSocketStream
from .notification_broker.local_broker import LocalNotificationBroker

# Global instances
logger = logging.getLogger(__name__)
notification_broker = LocalNotificationBroker()


class BasicHTTPException(HTTPException):
    """Custom HTTP exception that includes JSON data"""

    def __init__(self, status_code: int, data: Any):
        super().__init__(status_code=status_code)
        self.data = data


def basic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle BasicHTTPException and return JSON response"""
    if isinstance(exc, BasicHTTPException):
        return JSONResponse(content=exc.data, status_code=exc.status_code)
    return JSONResponse(
        content={"error": str(exc), "traceback": traceback.format_exc()}, status_code=500
    )


def get_project_id(headers: dict[str, str]) -> str | None:
    """Extract project ID from headers, return none if missing"""
    return headers.get("x-projectid")


async def invalidation_websocket(websocket: WebSocket):
    """Handle WebSocket connections for invalidations"""
    await websocket.accept()
    subscriber = None

    try:
        project_id = get_project_id(dict(websocket.headers))
        assert project_id is not None, "Project ID is required"
        subscriber = await notification_broker.subscribe(project_id, WebSocketStream(websocket))
        await subscriber.listen_for_messages()
    except BasicHTTPException as e:
        await websocket.close(code=400, reason=json.dumps(e.data))
    except Exception as e:
        logger.exception("Uncaught exception ")
        await websocket.close(code=500, reason=f"Internal server error: {str(e)}")
    finally:
        # Clean up the subscription
        if subscriber:
            await notification_broker.unsubscribe(subscriber)


def prepare_endpoint(func):
    """A decorator that abstracts the boilerplate needed to adapt the http data
    the agnostic daemon implementation."""

    @wraps(func)
    async def wrapper(request: Request) -> JSONResponse:
        project_id = get_project_id(dict(request.headers))
        # Prepare the data dictionary with the common attributes we extract
        # from the headers
        data = {
            "project_id": project_id,
            # TODO add auth token forwarding
        }
        response = await func(request, data)
        # Forward any websocket notification
        for notification in response.notifications:
            assert project_id is not None, "Project ID is required"
            await notification_broker.notify(project_id, json.dumps(notification))
        # Convert the daemon response to a JSON response
        return JSONResponse(
            status_code=response.code,
            content=response.body,
            headers=response.headers,
        )

    return wrapper


def make_starlette(daemon: Daemon) -> Starlette:
    # This doesn't use `prepare_endpoint` as it doesn't need the project_id nor token
    async def pipeline_endpoint(request: Request) -> JSONResponse:
        """Get pipeline information"""
        response = daemon.get_pipeline()
        return JSONResponse(
            status_code=response.code,
            content=response.body,
            headers=response.headers,
        )

    @prepare_endpoint
    async def epoch_endpoint(request: Request, data: dict) -> Response:
        """Get epoch information for a project"""
        return await daemon.get_epoch(data)

    @prepare_endpoint
    async def model_endpoint(request: Request, data: dict) -> Response:
        """Get model data for a project"""
        return await daemon.get_model(data)

    @prepare_endpoint
    async def artifact_endpoint(request: Request, data: dict) -> Response:
        """Process artifact requests"""
        return await daemon.artifact({**await request.json(), **data})

    @prepare_endpoint
    async def analysis_endpoint(request: Request, data: dict) -> Response:
        """Process analysis requests"""
        return await daemon.analyze({**await request.json(), **data})

    # Define routes
    routes = [
        Route("/api/epoch", epoch_endpoint, methods=["GET"]),
        Route("/api/pipeline", pipeline_endpoint, methods=["GET"]),
        Route("/api/model", model_endpoint, methods=["GET"]),
        Route("/api/artifact", artifact_endpoint, methods=["POST"]),
        Route("/api/analysis", analysis_endpoint, methods=["POST"]),
        WebSocketRoute("/api/subscribe", invalidation_websocket),
    ]

    origins: list[str] = []
    if "REVNG_ORIGINS" in os.environ:
        origins = os.environ["REVNG_ORIGINS"].split(",")

    expose_headers: list[str] = []
    if "REVNG_EXPOSE_HEADERS" in os.environ:
        expose_headers = os.environ["REVNG_EXPOSE_HEADERS"].split(",")

    # Create the Starlette application
    return Starlette(
        debug=False,
        routes=routes,
        exception_handlers={
            BasicHTTPException: basic_exception_handler,
        },
        middleware=[
            Middleware(  # type: ignore
                CORSMiddleware,  # type: ignore
                allow_origins=origins,
                expose_headers=expose_headers,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
            Middleware(GZipMiddleware, minimum_size=1024),  # type: ignore
        ],
    )
