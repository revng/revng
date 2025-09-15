#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass

from starlette.websockets import WebSocket, WebSocketDisconnect

from .utils import ProjectID

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message to be sent to subscribers"""
    project_id: ProjectID
    content: str


class WebSocketSubscriber:
    """Represents a WebSocket subscriber with its own message queue"""

    def __init__(self, project_id: ProjectID, websocket: WebSocket):
        self.project_id = project_id
        self.websocket = websocket
        self.message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.is_active = True

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

                try:
                    await self.websocket.send_text(message)
                except WebSocketDisconnect:
                    logger.debug(f"WebSocket disconnected for project {self.project_id}")
                    break
                except Exception as e:
                    logger.error(f"Error sending message to WebSocket for project {self.project_id}: {e}")
                    break
        except Exception as e:
            logger.error(f"Error in message listener for project {self.project_id}: {e}")
        finally:
            self.is_active = False

    def close(self):
        """Mark this subscriber as inactive"""
        self.is_active = False


class PostOffice:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.subscribers: dict[ProjectID, set[WebSocketSubscriber]] = defaultdict(set)

    async def subscribe(self, project_id: ProjectID, websocket: WebSocket) -> WebSocketSubscriber:
        """Subscribe a WebSocket to notifications for a project"""
        subscriber = WebSocketSubscriber(project_id, websocket)

        async with self._lock:
            self.subscribers[project_id].add(subscriber)

        logger.debug(f"WebSocket subscribed to project {project_id}")
        return subscriber

    async def unsubscribe(self, subscriber: WebSocketSubscriber):
        """Remove a WebSocket subscriber"""
        async with self._lock:
            project_id = subscriber.project_id
            if project_id in self.subscribers:
                self.subscribers[project_id].discard(subscriber)

                # Remove empty set to keep the dict clean
                if not self.subscribers[project_id]:
                    del self.subscribers[project_id]

        subscriber.close()
        logger.debug(f"WebSocket unsubscribed from project {subscriber.project_id}")

    async def notify(self, project_id: ProjectID, message: str | bytes):
        """Send a notification to all subscribers of a project"""
        if isinstance(message, bytes):
            message = message.decode('utf-8')

        async with self._lock:
            if project_id not in self.subscribers:
                logger.debug(f"No subscribers for project {project_id}")
                return

            # Get a copy of the subscribers set to avoid modification during iteration
            subscribers = self.subscribers[project_id].copy()

        logger.debug(f"Notifying {len(subscribers)} subscribers for project {project_id}")

        # Send messages to all active subscribers
        inactive_subscribers = []
        for subscriber in subscribers:
            if subscriber.is_active:
                try:
                    await subscriber.send_message(message)
                except Exception as e:
                    logger.error(f"Error queueing message for subscriber: {e}")
                    inactive_subscribers.append(subscriber)
            else:
                inactive_subscribers.append(subscriber)

        # Clean up inactive subscribers
        if inactive_subscribers:
            async with self._lock:
                if project_id in self.subscribers:
                    for subscriber in inactive_subscribers:
                        self.subscribers[project_id].discard(subscriber)

                    # Remove empty set
                    if not self.subscribers[project_id]:
                        del self.subscribers[project_id]

    async def get_subscriber_count(self, project_id: ProjectID) -> int:
        """Get the number of active subscribers for a project"""
        async with self._lock:
            if project_id not in self.subscribers:
                return 0
            return len([s for s in self.subscribers[project_id] if s.is_active])

    async def get_total_subscribers(self) -> int:
        """Get the total number of active subscribers across all projects"""
        async with self._lock:
            total = 0
            for subscribers in self.subscribers.values():
                total += len([s for s in subscribers if s.is_active])
            return total
