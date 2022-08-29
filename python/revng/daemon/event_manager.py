#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import os
import re
import signal
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, List, Protocol

from starlette.requests import Headers

from revng.api import Manager


class EventType(Enum):
    BEGIN = auto()
    CONTEXT = auto()


def emit_event(event_type: EventType):
    def decorator(f):
        @wraps(f)
        async def inner(*args, **kwargs):
            result = await f(*args, **kwargs)
            context = args[1].context
            if "event_manager" in context:
                context["event_manager"].handle_event(event_type, context["headers"])
            return result

        return inner

    return decorator


class EventHandler(Protocol):
    def __call__(self, manager: Manager, event_type: EventType, token: str):
        ...


class EventManager:
    def __init__(self, manager: Manager):
        self.listeners: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.manager = manager
        self.executor = ThreadPoolExecutor()
        if "REVNG_NOTIFY_FIFOS" in os.environ:
            self.initialize_fifos(os.environ["REVNG_NOTIFY_FIFOS"])

    def initialize_fifos(self, fifo_string: str):
        for fifo_definition in fifo_string.split(","):
            fifo_path, string_type = fifo_definition.split(":", 1)
            if string_type == "begin":
                self.add_event_listener(EventType.BEGIN, BeginListener(Path(fifo_path)))
            elif string_type == "context":
                self.add_event_listener(EventType.CONTEXT, ContextListener(Path(fifo_path)))
            else:
                logging.warn(f"Unknown event type specified for fifo: {string_type}")

    def add_event_listener(self, type_: EventType, callback: EventHandler):
        self.listeners[type_].append(callback)

    def handle_event(self, type_: EventType, headers: Headers):
        if type_ not in self.listeners:
            return

        def handler():
            for listener in self.listeners[type_]:
                listener(self.manager, type_, self.get_token(headers))

        # Run the callbacks in an executor to avoid blocking the result
        result = self.executor.submit(handler)
        result.add_done_callback(self.handle_event_result)

    @staticmethod
    def get_token(headers: Headers) -> str:
        if "authorization" not in headers or not headers["authorization"].startswith("Bearer "):
            return ""
        token = headers["authorization"][len("Bearer ") :]
        if re.search(r"\s", token):
            return ""
        return token

    @staticmethod
    def handle_event_result(result_future: Future):
        ex = result_future.exception()
        if ex is not None:
            logging.error("Exception raised while handling event callbacks", exc_info=ex)
            signal.raise_signal(signal.SIGINT)


class BaseListener(ABC, EventHandler):
    def __init__(self, fifo_path: Path):
        self.fifo = open(fifo_path, "w")  # noqa: SIM115

    @abstractmethod
    def __call__(self, manager: Manager, type_: EventType, token: str):
        pass

    def write_fifo(self, content: str):
        try:
            self.fifo.write(content)
            self.fifo.flush()
        except (BrokenPipeError, FileNotFoundError, PermissionError, TimeoutError) as e:
            logging.warn(
                f"Encountered error when writing to notification pipe: {repr(e)}",
            )

    def __del__(self):
        self.fifo.close()


class BeginListener(BaseListener):
    def __call__(self, manager: Manager, type_: EventType, token: str):
        assert type_ == EventType.BEGIN
        begin_step = manager.get_step("begin")
        assert begin_step is not None

        tmpdir = mkdtemp()
        begin_step.save(tmpdir)
        self.write_fifo(f"PUSH begin {tmpdir} {token}\n")


class ContextListener(BaseListener):
    def __call__(self, manager: Manager, type_: EventType, token: str):
        assert type_ == EventType.CONTEXT
        tmpdir = mkdtemp()
        manager.save_context(tmpdir)
        self.write_fifo(f"PUSH context {tmpdir} {token}\n")
