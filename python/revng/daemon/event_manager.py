#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import time
from enum import StrEnum, auto
from functools import wraps
from threading import Thread
from typing import Callable, Iterable

from revng.api import Manager


class EventType(StrEnum):
    BEGIN = auto()
    CONTEXT = auto()


def emit_event(event_type: EventType):
    """Decorator for allowing a GraphQL handler to emit an event after handling.
    This is expected to wrap handler-like function where the second argument is
    GraphQL's info object"""

    def decorator(f):
        @wraps(f)
        async def inner(*args, **kwargs):
            result = await f(*args, **kwargs)
            context = args[1].context
            if "event_manager" in context:
                context["event_manager"].handle_event(event_type)
            return result

        return inner

    return decorator


class EventManager(Thread):
    def __init__(self, manager: Manager, save_hooks: Iterable[Callable[[Manager, str], None]]):
        super().__init__()
        self.manager = manager
        self.save_hooks = save_hooks
        self.running = True
        self.next_save: float | None = None
        self.credentials: str | None = None

    def run(self):
        while self.running:
            # This loop will call manager.save after 5 minutes of inactivity. This is to prevent
            # the manager from lagging on every change. Some changes related to speeding up the
            # pipeline manager from saving (e.g. detecting dirtiness in containers) need to be
            # implemented before this can be dropped
            if self.next_save is not None and self.next_save < time.time():
                self.save()
            time.sleep(10)

    def save(self):
        result = self.manager.save()
        if result:
            self.next_save = None
            for hook in self.save_hooks:
                hook(self.manager, self.credentials)
        return result

    def handle_event(self, type_: EventType):
        self.next_save = time.time() + 300

    def set_credentials(self, credentials: str):
        self.credentials = credentials
        self.manager.set_storage_credentials(credentials)
