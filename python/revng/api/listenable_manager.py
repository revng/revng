#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import signal
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum, auto
from typing import Any, Callable, Dict, List

from .manager import Manager


class EventType(Enum):
    BEGIN = auto()
    CONTEXT = auto()


EventHandler = Callable[[Manager, EventType], Any]


class ListenableManager(Manager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._listeners: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._executor = ThreadPoolExecutor()

    def add_event_listener(self, type_: EventType, callback: EventHandler):
        self._listeners[type_].append(callback)

    def handle_event(self, type_: EventType):
        def handler():
            if type_ in self._listeners:
                for listener in self._listeners[type_]:
                    listener(self, type_)

        # Run the callbacks in an executor to avoid blocking the result
        result = self._executor.submit(handler)
        result.add_done_callback(self.handle_event_result)

    @staticmethod
    def handle_event_result(result_future: Future):
        ex = result_future.exception()
        if ex is not None:
            logging.error("Exception raised while handling event callbacks", exc_info=ex)
            signal.raise_signal(signal.SIGINT)

    def set_input(self, *args, **kwargs):
        result = super().set_input(*args, **kwargs)
        self.handle_event(EventType.BEGIN)
        return result

    def run_analysis(self, *args, **kwargs):
        result = super().run_analysis(*args, **kwargs)
        self.handle_event(EventType.CONTEXT)
        return result

    def run_all_analyses(self, *args, **kwargs):
        result = super().run_all_analyses()
        self.handle_event(EventType.CONTEXT)
        return result
