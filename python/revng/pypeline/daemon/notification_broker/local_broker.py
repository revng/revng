#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from revng.pypeline.storage.storage_provider import ProjectID

from . import NotificationBroker, NotificationPublisher, NotificationSubscriber, Stream

logger = logging.getLogger(__name__)


class LocalNotificationPublisher(NotificationPublisher):
    def __init__(self, broker: LocalNotificationBroker):
        self.broker = broker

    async def notify(self, project_id: ProjectID, message: str | bytes):
        await self.broker.notify(project_id, message)


class LocalNotificationBroker(NotificationBroker):
    def __init__(self):
        self._lock = asyncio.Lock()
        self.subscribers: dict[ProjectID, set[NotificationSubscriber]] = defaultdict(set)

    async def subscribe(self, project_id: ProjectID, stream: Stream) -> NotificationSubscriber:
        subscriber = NotificationSubscriber(project_id, stream)

        async with self._lock:
            self.subscribers[project_id].add(subscriber)

        logger.debug(f"Stream subscribed to project {project_id}")
        return subscriber

    async def unsubscribe(self, subscriber: NotificationSubscriber):
        async with self._lock:
            project_id = subscriber.project_id
            if project_id in self.subscribers:
                self.subscribers[project_id].discard(subscriber)

                # Remove empty set to keep the dict clean
                if len(self.subscribers[project_id]) == 0:
                    del self.subscribers[project_id]

        subscriber.close()
        logger.debug(f"Stream unsubscribed from project {subscriber.project_id}")

    def get_publisher(self) -> NotificationPublisher:
        return LocalNotificationPublisher(self)

    async def notify(self, project_id: ProjectID, message: str | bytes):
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        async with self._lock:
            if project_id not in self.subscribers:
                logger.debug(f"No subscribers for project {project_id}")
                return

            # Get a copy of the subscribers set to avoid modification during iteration
            subscribers = self.subscribers[project_id].copy()

        logger.debug(f"Notifying {len(subscribers)} subscribers for project {project_id}")

        # Send messages to all active subscribers
        inactive_subscribers: set[NotificationSubscriber] = set()
        for subscriber in subscribers:
            if subscriber.is_active:
                try:
                    await subscriber.send_message(message)
                except Exception as e:
                    logger.error(f"Error queueing message for subscriber: {e}")
                    inactive_subscribers.add(subscriber)
            else:
                inactive_subscribers.add(subscriber)

        # Clean up inactive subscribers
        if inactive_subscribers:
            async with self._lock:
                if project_id in self.subscribers:
                    for subscriber in inactive_subscribers:
                        self.subscribers[project_id].discard(subscriber)

                    # Remove empty set
                    if not self.subscribers[project_id]:
                        del self.subscribers[project_id]
