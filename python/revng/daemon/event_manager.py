#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import os
import re
import signal
from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum, auto
from functools import wraps
from io import TextIOWrapper
from tempfile import mkdtemp
from typing import Dict, List, Protocol

from starlette.requests import Headers

from revng.api import Manager


class EventType(StrEnum):
    BEGIN = auto()
    CONTEXT = auto()


class EventHandler(Protocol):
    def __call__(self, manager: Manager, event_type: EventType, token: str):
        ...


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
                context["event_manager"].handle_event(event_type, context["headers"])
            return result

        return inner

    return decorator


class EventManager:
    cls_handlers: List[EventHandler] = []

    def __init__(self, manager: Manager):
        self.handlers: List[EventHandler] = []
        self.manager = manager
        self.executor = ThreadPoolExecutor()

    @classmethod
    def add_class_handler(cls, handler: EventHandler):
        cls.cls_handlers.append(handler)

    def add_handler(self, handler: EventHandler):
        self.handlers.append(handler)

    def handle_event(self, type_: EventType, headers: Headers):
        def handler():
            for listener in self.handlers:
                listener(self.manager, type_, self.get_token(headers))
            for listener in self.cls_handlers:
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


class FIFOListener(EventHandler):
    def __init__(self):
        self.fifos: Dict[str, List[TextIOWrapper]] = {"begin": [], "context": []}
        if "REVNG_NOTIFY_FIFOS" in os.environ:
            fifo_string = os.environ["REVNG_NOTIFY_FIFOS"]
            for fifo_definition in fifo_string.split(","):
                fifo_path, fifo_type = fifo_definition.rsplit(":", 1)
                if fifo_type in self.fifos:
                    self.fifos[fifo_type].append(open(fifo_path, "w"))  # noqa: SIM115

    def __call__(self, manager: Manager, type_: EventType, token: str):
        if type_ == EventType.BEGIN:
            begin_step = manager.get_step("begin")
            assert begin_step is not None

            tmpdir = mkdtemp()
            begin_step.save(tmpdir)
            self.write_fifos(type_, f"PUSH begin {tmpdir} {token}\n")

        elif type_ == EventType.CONTEXT:
            tmpdir = mkdtemp()
            manager.save_context(tmpdir)
            self.write_fifos(type_, f"PUSH context {tmpdir} {token}\n")

    def write_fifos(self, type_: EventType, content: str):
        if str(type_) not in self.fifos:
            return

        for fifo in self.fifos[str(type_)]:
            try:
                fifo.write(content)
                fifo.flush()
            except (BrokenPipeError, FileNotFoundError, PermissionError, TimeoutError) as e:
                logging.warn(
                    f"Encountered error when writing to notification pipe: {repr(e)}",
                )

    def __del__(self):
        for fifos in self.fifos.values():
            for fifo in fifos:
                fifo.close()


EventManager.add_class_handler(FIFOListener())
