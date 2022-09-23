#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

from revng.api.listenable_manager import EventType, ListenableManager
from revng.api.manager import Manager
from revng.support import AnyPath


class BaseListener(ABC):
    def __init__(self, fifo_path: Path):
        self.fifo = open(fifo_path, "w")  # noqa: SIM115

    @abstractmethod
    def __call__(self, manager: Manager, type_: EventType):
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
    def __call__(self, manager: Manager, type_: EventType):
        assert type_ == EventType.BEGIN
        begin_step = manager.get_step("begin")
        assert begin_step is not None

        tmpdir = mkdtemp(prefix="revng-daemon-begin-")
        begin_step.save(tmpdir)
        self.write_fifo(f"PUSH begin {tmpdir}\n")


class ContextListener(BaseListener):
    def __call__(self, manager: Manager, type_: EventType):
        assert type_ == EventType.CONTEXT
        tmpdir = mkdtemp(prefix="revng-daemon-context-")
        manager.save_context(tmpdir)
        self.write_fifo(f"PUSH context {tmpdir}\n")


def make_manager(workdir: Optional[AnyPath]) -> Manager:
    if "REVNG_NOTIFY_FIFOS" not in os.environ:
        return Manager(workdir)

    manager = ListenableManager(workdir)
    for fifo_definition in os.environ["REVNG_NOTIFY_FIFOS"].split(","):
        fifo_path, string_type = fifo_definition.split(":", 1)
        if string_type == "begin":
            manager.add_event_listener(EventType.BEGIN, BeginListener(Path(fifo_path)))
        elif string_type == "context":
            manager.add_event_listener(EventType.CONTEXT, ContextListener(Path(fifo_path)))
        else:
            logging.warn(f"Unknown event type specified for fifo: {string_type}")

    return manager
