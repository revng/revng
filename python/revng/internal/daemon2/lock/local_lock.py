#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
from time import time
from typing import Any
from dataclasses import dataclass
from collections import defaultdict

from . import Lock, AlreadyLockedError, WrongEpoch, LockNotFoundError, LockExpiredError
from revng.internal.daemon2.utils import ProjectID, Epoch

@dataclass(slots=True)
class LockInfo:
    expiration: float
    metadata: Any

class LocalLock(Lock):
    """
    An in-memory lock useful if there is only one daemon running.
    """

    def __init__(self, lock_duration: int = 60):
        self._epochs: dict[ProjectID, Epoch] = defaultdict(lambda: Epoch(0))
        self._locks: dict[ProjectID, LockInfo] = {}
        self._locks_lock = asyncio.Lock()  # Async lock to protect access to self._locks
        self._lock_duration = lock_duration

    def get_epoch(self, project_id: ProjectID) -> Epoch:
        return self._epochs[project_id]

    async def lock(self, project_id: ProjectID, epoch: Epoch, metadata: Any = None) -> float:
        async with self._locks_lock:
            if project_id in self._locks:
                lock_info = self._locks[project_id]
                if time() < lock_info.expiration:
                    raise AlreadyLockedError(project_id, lock_info.metadata)
            if self._epochs[project_id] != epoch:
                raise WrongEpoch(project_id, self._epochs[project_id], epoch)
            expiration = time() + self._lock_duration
            self._locks[project_id] = LockInfo(expiration, metadata)
            return self._lock_duration

    async def renew_lock(self, project_id: ProjectID) -> float:
        async with self._locks_lock:
            if project_id not in self._locks:
                raise LockNotFoundError(project_id)
            lock_info = self._locks[project_id]
            if time() > lock_info.expiration:
                raise LockExpiredError(project_id, lock_info.expiration, lock_info.metadata)
            self._locks[project_id].expiration = time() + self._lock_duration
            return self._lock_duration

    async def unlock(self, project_id: ProjectID, increase_epoch: bool = True) -> Epoch:
        async with self._locks_lock:
            if project_id not in self._locks:
                raise LockNotFoundError(project_id)
            lock_info = self._locks.pop(project_id)
            if time() > lock_info.expiration:
                raise LockExpiredError(project_id, lock_info.expiration, lock_info.metadata)
            if increase_epoch:
                new_epoch = self._epochs[project_id] + 1
                self._epochs[project_id] = new_epoch
            return self._epochs[project_id]
