#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
from asyncio import Queue
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import override

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from websockets.frames import CloseCode

from revng.pypeline.storage.notification_queue import MultiQueue
from revng.pypeline.storage.storage_provider import ProjectID
from revng.pypeline.utils.notification_broker import NotificationBroker, NotificationSubscriber
from revng.pypeline.utils.notification_broker import Stream
from revng.pypeline.utils.starlette import NotificationWebsocket, get_middlewares, get_project_id

shutdown_begun: asyncio.Event | None = None


class MultiNotificationBroker(NotificationBroker):
    def __init__(self, queues: dict[ProjectID, MultiQueue[bytes]]):
        super().__init__()
        self.queues = queues

    @override
    async def subscribe(
        self, project_id: ProjectID | None, stream: Stream
    ) -> NotificationSubscriber:
        assert project_id is not None, "project_id = None is unsupported"
        return await super().subscribe(project_id, stream)

    @override
    async def get_queue(self, project_id: ProjectID | None) -> Queue[bytes]:
        assert project_id is not None
        return self.queues[project_id].get_queue()


def _parse_bind(string: str) -> tuple[str, int]:
    # TODO: fix for IPv6
    assert string.count(":") == 1
    parts = string.split(":")
    return (parts[0], int(parts[1]))


def _compare_connection(bind: tuple[str, int], connection: tuple[str, int]):
    """
    Given a bind address, e.g. `("127.0.0.1", 8000)` check that the connection
    address matches
    """

    # TODO: fix for IPv6
    if bind[0] == "0.0.0.0":
        # If the server is bound on all interfaces just check that the port
        # matches
        return bind[1] == connection[1]
    else:
        return bind == connection


def make_starlette(
    production: bool, notifications_bind: str, publish_bind: str, publish_psk: str
) -> Starlette:
    queues: dict[ProjectID, MultiQueue[bytes]] = defaultdict(MultiQueue)
    notification_broker = MultiNotificationBroker(queues)
    ws_notifications = NotificationWebsocket(notification_broker, lambda: shutdown_begun)

    notifications_tuple = _parse_bind(notifications_bind)
    publish_tuple = _parse_bind(publish_bind)

    async def status(request):
        return PlainTextResponse("OK")

    async def publish(request: Request):
        if not _compare_connection(publish_tuple, request["server"]):
            return PlainTextResponse(status_code=403)

        project_id = get_project_id(request.headers)
        if project_id is None:
            return PlainTextResponse("Project ID must be supplied", 400)

        # NOTE: at some point this static PSK will be dropped in favor of
        #       mTLS, but it has a high upfront infrastructural cost
        authorization = request.headers.get("authorization")
        if authorization != f"Bearer {publish_psk}":
            return PlainTextResponse("Invalid authorization header", 403)

        body = await request.body()
        queues[project_id].send(body)
        return PlainTextResponse("Sent")

    async def notifications(websocket: WebSocket):
        if not _compare_connection(notifications_tuple, websocket["server"]):
            await websocket.close(CloseCode.POLICY_VIOLATION)
            return
        return await ws_notifications.endpoint(websocket)

    @asynccontextmanager
    async def lifespan(app):
        global shutdown_begun
        shutdown_begun = asyncio.Event()
        yield

    # Create the Starlette application
    return Starlette(
        debug=not production,
        routes=[
            Route("/publish", publish, methods=["POST"]),
            WebSocketRoute("/notifications", notifications),
            Route("/status", status, methods=["GET"]),
        ],
        middleware=get_middlewares(production, unauthenticated_paths={"/publish"}),
        lifespan=lifespan,
    )
