#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
from contextlib import asynccontextmanager
from functools import wraps
from typing import Callable

from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.responses import Response as StarletteResponse
from starlette.routing import BaseRoute, Route, WebSocketRoute

from revng.pypeline.storage.notification_queue import LOCAL_QUEUE
from revng.pypeline.storage.storage_provider import ProjectID
from revng.pypeline.utils import PypelineException
from revng.pypeline.utils.notification_broker import NotificationBroker
from revng.pypeline.utils.starlette import NotificationWebsocket, get_middlewares, get_project_id
from revng.pypeline.utils.starlette import get_token

from .daemon import Daemon, Response
from .exceptions import DaemonException, MalformedRequestError

# Global instances


class LocalNotificationBroker(NotificationBroker):
    async def get_queue(self, project_id: ProjectID | None) -> asyncio.Queue[bytes]:
        return LOCAL_QUEUE.get_queue()


notification_broker = LocalNotificationBroker()
# This is initialized by the `lifespan` function below, once the actual event
# loop has been created, otherwise this creates another event loop and an
# exception is thrown since the two loops don't match.
shutdown_begun: asyncio.Event | None = None


def daemon_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DaemonException)
    """Handle BasicHTTPException and return JSON response"""
    return JSONResponse(content=exc.body, status_code=exc.code)


def pypeline_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, PypelineException)
    """Handle BasicHTTPException and return JSON response"""
    return JSONResponse(content={"message": str(exc)}, status_code=500)


def prepare_endpoint(func):
    """A decorator that abstracts the boilerplate needed to adapt the http data
    the agnostic daemon implementation."""

    @wraps(func)
    async def wrapper(request: Request) -> StarletteResponse:
        # Prepare the data dictionary with the common attributes we extract
        # from the headers
        data = {
            "project_id": get_project_id(request.headers),
            "token": get_token(request.headers),
        }
        response: Response = await func(request, data)
        # Convert the daemon response based on the body type:
        # * a JSON response if the body is a dict or list
        # * a binary response if the body is bytes
        # * throw an error for everything else
        if isinstance(response.body, (dict, list)):
            return JSONResponse(
                status_code=response.code,
                content=response.body,
                headers=response.headers,
            )
        elif isinstance(response.body, bytes):
            return StarletteResponse(
                status_code=response.code,
                media_type=response.content_type,
                content=response.body,
                headers=response.headers,
            )
        else:
            raise ValueError(f"Unknown response body type: {type(response.body)}")

    return wrapper


def make_starlette(
    production: bool,
    daemon: Daemon,
    on_startup: Callable[[], None] = lambda: None,
) -> Starlette:
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
    async def put_file_endpoint(request: Request, data: dict) -> Response:
        """Put a file in storage"""
        async with request.form() as form:
            if not isinstance(form["file"], UploadFile):
                raise MalformedRequestError('"file" parameter is not a file')
            file: UploadFile = form["file"]
            return await daemon.put_file(
                {"name": file.filename, "contents": await file.read(), **data}
            )

    @prepare_endpoint
    async def artifact_endpoint(request: Request, data: dict) -> Response:
        """Process artifact requests"""
        return await daemon.artifact({**await request.json(), **data})

    @prepare_endpoint
    async def analysis_endpoint(request: Request, data: dict) -> Response:
        """Process analysis requests"""
        return await daemon.analyze({**await request.json(), **data})

    async def status(request):
        return PlainTextResponse("OK")

    # Define routes
    routes: list[BaseRoute] = [
        Route("/api/epoch", epoch_endpoint, methods=["GET"]),
        Route("/api/pipeline", pipeline_endpoint, methods=["GET"]),
        Route("/api/model", model_endpoint, methods=["GET"]),
        Route("/api/put-file", put_file_endpoint, methods=["POST"]),
        Route("/api/artifact", artifact_endpoint, methods=["POST"]),
        Route("/api/analysis", analysis_endpoint, methods=["POST"]),
        Route("/status", status, methods=["GET"]),
    ]

    websocket_url = daemon.storage_provider_factory.get_notification_websocket()
    if websocket_url is not None:

        async def websocket_url_handler(request: Request):
            return JSONResponse({"url": websocket_url})

        routes.append(Route("/api/websocket-url", websocket_url_handler, methods=["GET"]))
    else:

        async def websocket_url_handler(request: Request):
            return JSONResponse({"url": "/api/notifications"})

        routes.append(Route("/api/websocket-url", websocket_url_handler, methods=["GET"]))
        ws_notifications = NotificationWebsocket(notification_broker, lambda: shutdown_begun)
        routes.append(WebSocketRoute("/api/notifications", ws_notifications.endpoint))

    @asynccontextmanager
    async def lifespan(app):
        on_startup()
        global shutdown_begun
        shutdown_begun = asyncio.Event()
        yield

    # Create the Starlette application
    return Starlette(
        debug=not production,
        routes=routes,
        exception_handlers={
            DaemonException: daemon_exception_handler,
            PypelineException: pypeline_exception_handler,
        },
        middleware=get_middlewares(
            production,
            extra_expose_headers={"x-pypeline-configuration-hash"},
            unauthenticated_paths={"/api/websocket-url"},
        ),
        lifespan=lifespan,
    )
