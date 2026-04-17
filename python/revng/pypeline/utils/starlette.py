#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
from contextlib import suppress
from typing import Callable, Mapping

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
