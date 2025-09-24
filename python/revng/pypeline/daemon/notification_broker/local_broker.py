#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from revng.pypeline.storage.storage_provider import ProjectID
from revng.pypeline.utils import Locked

from . import NotificationBroker, NotificationSubscriber, Stream

logger = logging.getLogger(__name__)


class LocalNotificationBroker(NotificationBroker):
    def __init__(self):
        self._lock = asyncio.Lock()
        self.subscribers: Locked[dict[ProjectID, Locked[set[NotificationSubscriber]]]] = Locked(
            defaultdict(lambda: Locked(set()))
        )

    async def subscribe(self, project_id: ProjectID, stream: Stream) -> NotificationSubscriber:
        subscriber = NotificationSubscriber(project_id, stream)

        async with self.subscribers() as subscribers:
            async with subscribers[project_id]() as project_subscribers:
                project_subscribers.add(subscriber)

        logger.debug(f"Stream subscribed to project {project_id}")
        return subscriber

    async def unsubscribe(self, subscriber: NotificationSubscriber):
        async with self.subscribers() as subscribers:
            project_id = subscriber.project_id
            async with subscribers[project_id]() as project_subscribers:
                if project_id in subscribers:
                    project_subscribers.discard(subscriber)

        subscriber.close()
        logger.debug(f"Stream unsubscribed from project {subscriber.project_id}")

    async def notify(self, project_id: ProjectID, message: str):
        async with self.subscribers() as subscribers:
            if project_id not in subscribers:
                logger.debug(f"No subscribers for project {project_id}")
                return

            # Get a copy of the subscribers set to avoid modification during iteration
            async with subscribers[project_id]() as project_subscribers:
                project_subscribers = project_subscribers.copy()

        logger.debug(f"Notifying {len(project_subscribers)} subscribers for project {project_id}")

        # Send messages to all active subscribers
        inactive_subscribers: set[NotificationSubscriber] = set()
        for subscriber in project_subscribers:
            if subscriber.is_active:
                try:
                    await subscriber.send_message(message)
                except Exception as e:
                    logger.error(f"Error queueing message for subscriber: {e}")
                    inactive_subscribers.add(subscriber)
            else:
                inactive_subscribers.add(subscriber)

        # Clean up inactive subscribers
        if not inactive_subscribers:
            return
        async with self.subscribers() as subscribers:
            if project_id not in subscribers:
                return
            async with subscribers[project_id]() as project_subscribers:
                for subscriber in inactive_subscribers:
                    project_subscribers.discard(subscriber)
