#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Protocol

from starlette.websockets import WebSocket

from revng.pypeline.storage.storage_provider import ProjectID

logger = logging.getLogger(__name__)


class Stream(Protocol):
    """Represents a bidirectional stream, like a TCP socket or a WebSocket"""

    async def read(self) -> bytes: ...
    async def write(self, data: bytes) -> None: ...


class WebSocketStream(Stream):
    """Represents a WebSocket stream"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def read(self) -> bytes:
        return await self.websocket.receive_bytes()

    async def write(self, data: bytes) -> None:
        await self.websocket.send_bytes(data)


class NotificationSubscriber:
    """Represents a notification subscriber with its own message queue"""

    def __init__(self, project_id: ProjectID, stream: Stream):
        self.project_id = project_id
        self.stream = stream
        self.message_queue: asyncio.Queue[str] = asyncio.Queue()
        """The broker will insert messages into this queue, and the subscriber will
        listen for them and insert them into the message queue."""
        self.is_active = True
        """Flag indicating whether the subscriber is active, used so the broker
        can garbage collect inactive subscribers."""

    async def send_message(self, message: str):
        """Add a message to this subscriber's queue"""
        if self.is_active:
            await self.message_queue.put(message)

    async def listen_for_messages(self):
        """Listen for messages in the queue and send them via WebSocket"""
        try:
            while self.is_active:
                # Wait for a message in the queue
                message = await self.message_queue.get()
                if not self.is_active:
                    break

                await self.stream.write(message.encode())
        finally:
            self.is_active = False

    def close(self):
        """Mark this subscriber as inactive"""
        self.is_active = False


class NotificationBroker(ABC):
    """NotificationBroker is used to notify all subscribers of a project of changes.
    Our current intended uses are:
    - An in-process broker that sends notifications to subscribers via WebSocket.
    - A distributed broker implemented as a separate service using something like
        Pub/Sub.
    """

    @abstractmethod
    async def subscribe(self, project_id: ProjectID, stream: Stream) -> NotificationSubscriber:
        """Register to receive notifications for project \"project_id\" on the given stream"""

    @abstractmethod
    async def unsubscribe(self, subscriber: NotificationSubscriber):
        """Stop receiving notifications for project \"project_id\"."""

    @abstractmethod
    async def notify(self, project_id: ProjectID, message: str):
        """Notify all subscribers of project \"project_id\" of the given message"""
