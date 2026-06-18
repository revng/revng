#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import os
from contextlib import suppress
from importlib import import_module
from typing import Callable, Mapping

from starlette.datastructures import Headers, QueryParams
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect

from websockets.frames import CloseCode

from revng.pypeline.utils.logger import pypeline_logger
from revng.pypeline.utils.notification_broker import NotificationBroker, Stream


class WebSocketStream(Stream):
    """Represents a WebSocket stream"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def read(self) -> bytes:
        return await self.websocket.receive_bytes()

    async def write(self, data: bytes) -> None:
        await self.websocket.send_bytes(data)


def get_project_id(headers: Mapping[str, str]) -> str | None:
    """Extract project ID from headers, return None if missing"""
    return headers.get("x-project-id")


def get_token(headers: Mapping[str, str]) -> str | None:
    """
    Extract the token from the Authorization header, return None if the header
    is missing or it does not begin with `Bearer `.
    """
    if "authorization" in headers and headers["authorization"].startswith("Bearer "):
        return headers["authorization"][len("Bearer ") :]
    else:
        return None


class NotificationWebsocket:
    def __init__(
        self,
        notification_broker: NotificationBroker,
        shutdown_getter: Callable[[], asyncio.Event | None],
    ):
        self.notification_broker = notification_broker
        self.shutdown_getter = shutdown_getter

    async def endpoint(self, websocket: WebSocket):
        """Handle WebSocket connections for invalidations"""
        shutdown_begun = self.shutdown_getter()
        assert shutdown_begun is not None
        await websocket.accept()
        subscriber = None
        pending = None

        try:
            project_id = websocket.query_params.get("project_id")
            subscriber = await self.notification_broker.subscribe(
                project_id, WebSocketStream(websocket)
            )
            done, pending = await asyncio.wait(
                (
                    asyncio.create_task(shutdown_begun.wait()),
                    asyncio.create_task(subscriber.listen_for_messages()),
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
            # If here the `asyncio.wait` has finished, this means that:
            # * `done` contains exactly one task
            # * the task in `done` might have an exception
            # Iterate over it and raise the exception if present
            for done_task in done:
                if (exc := done_task.exception()) is not None:
                    raise exc
        except WebSocketDisconnect:
            pass
        except Exception as e:
            pypeline_logger.log(f"Uncaught exception: {str(e)}")
            with suppress(RuntimeError):
                await websocket.close(
                    code=CloseCode.INTERNAL_ERROR,
                    reason=f"Internal server error: {str(e)}",
                )
        finally:
            # Clean up unfinished tasks
            if pending is not None:
                for task in pending:
                    task.cancel()
            # Clean up the subscription
            if subscriber is not None:
                await self.notification_broker.unsubscribe(subscriber)


class AuthMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        production: bool = True,
        authenticator: str | None = None,
        unauthenticated_paths: set[str] = set(),
    ):
        self.app = app
        self.unauthenticated_paths = {"/status", *unauthenticated_paths}

        self.check_token: Callable[[str | None, str | None], bool]
        if authenticator is not None:
            package_string, function_string = authenticator.split(":")
            package = import_module(package_string)
            self.check_token = getattr(package, function_string)
        elif not production:
            # When not in production allow any token
            self.check_token = lambda x, y: True
        else:
            # When in production it is expected that a custom authenticator is
            # supplied, so to avoid misconfigurations the default is to deny
            # all tokens
            self.check_token = lambda x, y: False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            # Allow lifespan calls through
            return await self.app(scope, receive, send)

        elif scope["type"] == "http":
            # Handle HTTP connection, it is expected either:
            # * an authenticated path (e.g. `/status`), which is let through
            #   without authentication
            # * A request with the `Authorization: Bearer <token>` header

            # Let the health check endpoint and any other paths that have been
            # excluded through without auth
            if scope["path"] in self.unauthenticated_paths:
                return await self.app(scope, receive, send)

            # Extract the project_id and token from the headers
            headers = Headers(scope=scope)
            project_id = get_project_id(headers)
            token = get_token(headers)

            # Here the token is checked, if the token is valid then the
            # check_token function will return True
            if self.check_token(project_id, token):
                return await self.app(scope, receive, send)
            else:
                response = PlainTextResponse("Invalid Token", status_code=401)
                return await response(scope, receive, send)

        elif scope["type"] == "websocket":
            # Handle websocket connections, the connection URL will have the format:
            # `wss://<host>/<path>/?project_id=<project_id>&token=<token>`
            # TODO: the token should be sent directly to the websocket as a
            #       message, this complicates things but avoids encoding the
            #       token in the URL.

            # Extract the project_id and token from query parameters
            query_parameters = QueryParams(scope["query_string"])
            project_id = query_parameters.get("project_id")
            token = query_parameters.get("token")

            # Check the token
            if self.check_token(project_id, token):
                return await self.app(scope, receive, send)
            else:
                websocket = WebSocket(scope, receive, send)
                return await websocket.close(CloseCode.POLICY_VIOLATION, "Invalid Token")

        else:
            # Any other middleware should be rejected
            raise ValueError(f"Unknown scope type: {scope['type']}")


def get_middlewares(
    production: bool,
    *,
    extra_expose_headers: set[str] = set(),
    auth: bool = True,
    unauthenticated_paths: set[str] = set(),
) -> list[Middleware]:
    origins: list[str] = []
    if "PYPELINE_ORIGINS" in os.environ:
        origins = os.environ["PYPELINE_ORIGINS"].split(",")
    if not production:
        origins.append("*")

    expose_headers: list[str] = [*extra_expose_headers]
    if "PYPELINE_EXPOSE_HEADERS" in os.environ:
        expose_headers.extend(os.environ["PYPELINE_EXPOSE_HEADERS"].split(","))
    if auth:
        expose_headers.append("x-project-id")

    middlewares = [
        Middleware(  # type: ignore
            CORSMiddleware,  # type: ignore
            allow_origins=origins,
            expose_headers=expose_headers,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        Middleware(GZipMiddleware, minimum_size=1024),  # type: ignore
    ]
    if auth:
        middlewares.append(
            Middleware(
                AuthMiddleware,
                production=production,
                authenticator=os.environ.get("PYPELINE_AUTHENTICATOR"),
                unauthenticated_paths=unauthenticated_paths,
            )
        )

    return middlewares
