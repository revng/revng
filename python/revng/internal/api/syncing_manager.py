#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import enum
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory, mkdtemp
from threading import Event, Lock, Thread
from typing import Any, Callable, Iterable

from .manager import Manager
from .synchronizer import S3Synchronizer, Synchronizer

__all__ = ["SaveHook", "SaveHooks", "SyncingManager"]
SaveHook = Callable[[Manager, str], None]
SaveHooks = Iterable[SaveHook]


class StorageSaver(Thread):
    def __init__(self, manager: Manager, synchronizer: Synchronizer, save_hooks: SaveHooks):
        super().__init__()
        self.manager = manager
        self.synchronizer = synchronizer
        self.save_hooks = save_hooks
        # Setting this to true will signal the thread to stop working as soon
        # as possible
        self.stopping = False
        # This event will be set when the thread finishes running
        self.shutdown_event = Event()

        # All the variables after this are guarded by this lock. Reading
        # and/or modifying the variables requires acquiring the lock.
        self.lock = Lock()

        # This event is set when someone calls the `submit` method
        self.submit_event = Event()
        # List of directories submitted for upload. If more than one is present
        # there will be a single upload "transaction" and all the directories
        # will be deleted afterwards.
        self.directory_list: list[str] = []
        # Credentials captured by `set_credentials`, will be passed to
        # `save_hooks` when calling them
        self.credentials = synchronizer.get_initial_credentials()

    def more_work(self) -> bool:
        with self.lock:
            return not (len(self.directory_list) == 0 and self.stopping)

    def run(self):
        while self.more_work():
            result = self.submit_event.wait(timeout=10)
            if not result:
                continue

            with self.lock:
                old_dir_list = self.directory_list.copy()
                self.directory_list.clear()
                self.submit_event.clear()

            last_directory = Path(old_dir_list[-1])
            old_directories = [Path(p) for p in reversed(old_dir_list[:-1])]

            try:
                self.synchronizer.save(last_directory, old_directories)
            except self.synchronizer.save_exceptions() as e:
                sys.stderr.write("Failed syncing resume directory\n")
                traceback.print_exception(e, file=sys.stderr)
                sys.stderr.flush()

            for element in old_dir_list:
                rmtree(element, False)

            with self.lock:
                credentials = self.credentials

            for hook in self.save_hooks:
                hook(self.manager, credentials)

        self.shutdown_event.set()

    def submit(self, directory: str):
        """Submit a directory for upload, cannot be done after stopping"""
        assert not self.stopping, "Cannot submit while stopping"
        with self.lock:
            self.directory_list.append(directory)
            self.submit_event.set()

    def set_credentials(self, credentials: str):
        with self.lock:
            self.credentials = credentials
            self.synchronizer.set_credentials(credentials)


class ResettableTimer(Thread):
    def __init__(self, interval: int, function: Callable[[], Any]):
        super().__init__()
        self.interval = interval
        self.function = function
        self.running = True

        # This lock protects the `next_trigger_time variable`
        self.lock = Lock()
        self.next_trigger_time = None

    def run(self):
        while self.running:
            if self.next_trigger_time is not None and time.time() > self.next_trigger_time:
                self.reset_countdown()
                self.function()
            time.sleep(10)

    def reset_countdown(self):
        """Resets the countdown, the function will not be called"""
        with self.lock:
            self.next_trigger_time = None

    def start_countdown(self):
        """Start countdown, will trigger the execution of `function` after
        `interval` seconds have passed from the first `start_countdown`. The
        function will not be called if a `reset_countdown` happens in the
        meantime."""
        with self.lock:
            if self.next_trigger_time is None:
                self.next_trigger_time = time.time() + self.interval


class EventType(enum.StrEnum):
    VOLATILE_CHANGE = enum.auto()
    SAVE = enum.auto()


class SyncingManager(Manager):
    SAVE_INTERVAL = 120

    def __init__(
        self,
        workdir: str | None = None,
        flags: Iterable[str] = (),
        save_hooks: SaveHooks = (),
    ):
        synchronizer: Synchronizer | None = None
        self._storage_saver: StorageSaver | None = None

        if workdir is None:
            self._temporary_workdir = TemporaryDirectory(prefix="revng-manager-workdir-")
            workdir = self._temporary_workdir.name
        elif workdir.startswith("s3://") or workdir.startswith("s3s://"):
            synchronizer = S3Synchronizer(workdir)
            self._storage_saver = StorageSaver(self, synchronizer, save_hooks)
            self._storage_saver.start()

            self._temporary_workdir = TemporaryDirectory(prefix="revng-manager-workdir-")
            workdir = self._temporary_workdir.name

        if synchronizer is not None:
            synchronizer.load(Path(workdir))

        super().__init__(workdir, flags)
        # This lock needs to be taken every time the workdir needs to be moved
        # elsewhere, by using `_move_workdir_for_save`
        self.save_lock = Lock()
        self.workdir_path = Path(self.workdir)

        if self._storage_saver is not None:
            with self.save_lock:
                temp_dir = self._move_workdir_for_save()
            self._storage_saver.submit(temp_dir)

        # Create and start the timer thread that triggers saves
        self.save_timer = ResettableTimer(self.__class__.SAVE_INTERVAL, self.save)
        self.save_timer.start()

    def stop(self):
        """Stops the manager's threads and waits for their termination"""
        self.save_timer.running = False
        if self._storage_saver is not None:
            self._storage_saver.stopping = True
            self._storage_saver.shutdown_event.wait()
        self.save_timer.join()

    def _move_workdir_for_save(self) -> str:
        assert self.save_lock.locked()
        # Replace the supplied directory, using a temporary directory in the
        # same parent directory. The temporary directory is created in the
        # parent directory to mitigate crossing filesystem boundaries as
        # much as possible (otherwise the `os.replace` call below will
        # fail).
        temp_dir = mkdtemp(prefix="manager-synchronizer-", dir=self.workdir_path.parent)

        # Moves the directory to the temporary directory, atomically (if the
        # filesystem is POSIX compliant). Will fail if it cannot do so. After
        # this call files in `path` will be in the same place in the temporary
        # directory, e.g.
        # `<path.parent>/<path.name>/context/model.yml` ->
        # `<path.parent>/<temp_dir_path.name>/context/model.yml`
        os.replace(self.workdir, temp_dir)
        # Remake the workdir directory
        self.workdir_path.mkdir()

        return temp_dir

    @contextmanager
    def _storage_function_manager(self):
        if self._storage_saver:
            with self.save_lock:
                yield None
                temp_dir = self._move_workdir_for_save()
                self._storage_saver.submit(temp_dir)
        else:
            yield None
        self.handle_event(EventType.SAVE)

    def set_storage_credentials(self, credentials: str):
        if self._storage_saver is not None:
            self._storage_saver.set_credentials(credentials)

    def handle_event(self, event_type: EventType):
        if event_type == EventType.SAVE:
            self.save_timer.reset_countdown()
        elif event_type == EventType.VOLATILE_CHANGE:
            self.save_timer.start_countdown()

    #
    # All functions that follow are wrappers around functions in `Manager`
    #
    def set_input(self, *args, **kwargs):
        result = super().set_input(*args, **kwargs)
        self.handle_event(EventType.VOLATILE_CHANGE)
        return result

    def produce_target(self, *args, **kwargs):
        result = super().produce_target(*args, **kwargs)
        self.handle_event(EventType.VOLATILE_CHANGE)
        return result

    def save(self, *args, **kwargs):
        with self._storage_function_manager():
            return super().save(*args, **kwargs)

    def run_analysis(self, *args, **kwargs):
        with self._storage_function_manager():
            return super().run_analysis(*args, **kwargs)

    def run_analyses_list(self, *args, **kwargs):
        with self._storage_function_manager():
            return super().run_analyses_list(*args, **kwargs)
