#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from abc import ABC, abstractmethod
from asyncio import Queue
from collections import defaultdict
from typing import Protocol

from revng.pypeline.storage.storage_provider import ProjectID
from revng.pypeline.utils import Locked
from revng.pypeline.utils.logger import pypeline_logger


class Stream(Protocol):
    """Represents a bidirectional stream, like a TCP socket or a WebSocket"""

    async def read(self) -> bytes: ...
    async def write(self, data: bytes) -> None: ...


class NotificationSubscriber:
    """Represents a notification subscriber with its own message queue"""

    def __init__(self, project_id: ProjectID | None, stream: Stream, message_queue: Queue[bytes]):
        self.project_id = project_id
        self.stream = stream
        self.message_queue = message_queue
        """The broker will insert messages into this queue, and the subscriber will
        listen for them and insert them into the message queue."""
        self.is_active = True
        """Flag indicating whether the subscriber is active, used so the broker
        can garbage collect inactive subscribers."""

    async def listen_for_messages(self):
        """Listen for messages in the queue and send them via WebSocket"""
        try:
            while self.is_active:
                # Wait for a message in the queue
                message = await self.message_queue.get()
                if not self.is_active:
                    break

                await self.stream.write(message)
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

    def __init__(self):
        self.subscribers: Locked[dict[ProjectID | None, Locked[set[NotificationSubscriber]]]] = (
            Locked(defaultdict(lambda: Locked(set())))
        )

    async def subscribe(
        self, project_id: ProjectID | None, stream: Stream
    ) -> NotificationSubscriber:
        """Register to receive notifications for project \"project_id\" on the given stream"""
        queue = await self.get_queue(project_id)
        subscriber = NotificationSubscriber(project_id, stream, queue)

        async with self.subscribers() as subscribers:
            async with subscribers[project_id]() as project_subscribers:
                project_subscribers.add(subscriber)

        pypeline_logger.debug_log(f"Stream subscribed to project {project_id}")
        return subscriber

    async def unsubscribe(self, subscriber: NotificationSubscriber):
        """Stop receiving notifications for project \"project_id\"."""
        async with self.subscribers() as subscribers:
            project_id = subscriber.project_id
            async with subscribers[project_id]() as project_subscribers:
                if project_id in subscribers:
                    project_subscribers.discard(subscriber)

        subscriber.close()
        pypeline_logger.debug_log(f"Stream unsubscribed from project {subscriber.project_id}")

    @abstractmethod
    async def get_queue(self, project_id: ProjectID | None) -> Queue[bytes]:
        """
        Get the queue that the subscriber will receive messages in. This should
        be exclusive to the subscriber.
        """
