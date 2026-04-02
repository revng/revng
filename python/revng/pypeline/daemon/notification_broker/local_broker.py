#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from collections import defaultdict

from revng.pypeline.storage.storage_provider import ProjectID
from revng.pypeline.utils import Locked
from revng.pypeline.utils.logger import pypeline_logger

from . import NotificationBroker, NotificationSubscriber, Stream


class LocalNotificationBroker(NotificationBroker):
    def __init__(self):
        self.subscribers: Locked[dict[ProjectID | None, Locked[set[NotificationSubscriber]]]] = (
            Locked(defaultdict(lambda: Locked(set())))
        )

    async def subscribe(
        self, project_id: ProjectID | None, stream: Stream
    ) -> NotificationSubscriber:
        subscriber = NotificationSubscriber(project_id, stream)

        async with self.subscribers() as subscribers:
            async with subscribers[project_id]() as project_subscribers:
                project_subscribers.add(subscriber)

        pypeline_logger.debug_log(f"Stream subscribed to project {project_id}")
        return subscriber

    async def unsubscribe(self, subscriber: NotificationSubscriber):
        async with self.subscribers() as subscribers:
            project_id = subscriber.project_id
            async with subscribers[project_id]() as project_subscribers:
                if project_id in subscribers:
                    project_subscribers.discard(subscriber)

        subscriber.close()
        pypeline_logger.debug_log(f"Stream unsubscribed from project {subscriber.project_id}")
