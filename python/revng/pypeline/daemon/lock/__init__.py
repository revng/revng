#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any

from revng.pypeline.daemon.utils import Epoch
from revng.pypeline.storage.storage_provider import ProjectID

logger = logging.getLogger(__name__)


class LockError(RuntimeError):
    """Base class for all lock-related exceptions."""


class AlreadyLockedError(LockError):
    """Raised when trying to lock a project that is already locked."""

    def __init__(self, project_id: ProjectID, metadata: Any = None):
        self.project_id = project_id
        self.metadata = metadata
        message = f"Project `{project_id}` is already locked."
        if metadata:
            message += f" Metadata: `{metadata}`"
        super().__init__(message)


class LockExpiredError(LockError):
    """Raised when trying to access a lock that has expired."""

    def __init__(self, project_id: ProjectID, expired: float, metadata: Any = None):
        self.project_id = project_id
        self.expired = expired
        self.metadata = metadata
        message = f"Lock for project `{project_id}` has expired."
        if expired:
            message += f" Expired at: `{expired}`"
        if metadata:
            message += f" Metadata: `{metadata}`"
        super().__init__(message)


class LockNotFoundError(LockError):
    """Raised when trying to access a lock that does not exist."""

    def __init__(self, project_id: ProjectID):
        self.project_id = project_id
        super().__init__(f"Lock for project `{project_id}` not found.")


class WrongEpoch(LockError):
    """Raised when trying to lock a project with an epoch that does not match the current one."""

    def __init__(self, project_id: ProjectID, expected_epoch: Epoch, actual_epoch: Epoch):
        self.project_id = project_id
        self.metadata = {"expected_epoch": expected_epoch, "actual_epoch": actual_epoch}
        super().__init__(
            f"Project `{project_id}` expected epoch `{expected_epoch}`, but got `{actual_epoch}`."
        )


class Lock(ABC):
    """
    Lock the access to epoch and model, the goal is that only one user at time can
    modify the model.

    We employ lock durations instead of an expiration date to avoid assuming that
    the system clock of the clients and servers are aligned.
    Using a duration introduces a constraint, the lock duration has to be at least
    two times the network Rount Trip Time, plus the time needed to renew the lock,
    plus the two computer clocks might run at slightly different speeds that will
    cause a drift between the effective durations.
    T_lease > T_rtt + T_renew + T_drift

    The effective duration for the client will be less than the returned value, as it will arrive
    after passing through the network.
    T_{client lease} < T_lease - T_rtt / 2
    Moreover, it will need another network passage, then the time needed by the server to process
    the renewal, and the possible clocks drift:
    T_{client lease} < T_lease - T_rtt - T_renew - T_drift

    To avoid having to measure T_rtt, T_renew, and T_drift it is reasonable to assume that:
    T_rtt < 2 seconds
    T_renew < 2 seconds
    T_drift < 1% T_lease
    and we shouldn't renew too fast to avoid wasting CPU time so:
    T_{client lease} > 4 seconds
    therefore a reasonable values for T_lease will be over ten seconds:
    T_lease > 10 seconds
    and the client should consider a safety margin to renew the lease:
    T_{client lease} = 75% T_lease
    """

    @abstractmethod
    def get_epoch(self, project_id: ProjectID) -> Epoch:
        """
        Return the current epoch for the project, without locking, as, if we get a data-race, the
        application will get a `WrongEpoch` exception with the correct new epoch.
        """

    @abstractmethod
    async def lock(self, project_id: ProjectID, epoch: Epoch, metadata: Any = None) -> float:
        """
        Attempts to acquire a project lock when the provided epoch matches the expected value.
        If the lock is already held, it will raise an `AlreadyLockedError` exception.
        Any provided metadata will be attached to any raised exception, their porpouse is to allow
        the caller to add metadata that will help tracking and correlating errors.
        Returns the lock duration in seconds, representing the time remaining before the lock
        expires and must be renewed, with `renew_lock`, to maintain ownership.
        """

    @abstractmethod
    async def renew_lock(self, project_id: ProjectID) -> float:
        """
        Renews an active project lock, extending its validity period.
        Returns the new lock duration in seconds, which is intentionally variable.
        The caller must invoke this method periodically before the current
        lock expires to maintain continuous ownership of the project lock.
        """

    @abstractmethod
    async def unlock(self, project_id: ProjectID, increase_epoch: bool = True) -> Epoch:
        """Release the lock and increase the epoch if needed."""


class LockManager:
    """An async context manager around a Lock so we don't forget to unlock it and it automatically
    renews the lock when needed."""

    def __init__(
        self,
        lock: Lock,
        project_id: ProjectID,
        epoch: Epoch,
        metadata: Any = None,
        increase_epoch: bool = True,
    ):
        """
        Arguments:
        - lock: The lock this context manager will try to acquire
        - project_id: The project we want to lock
        - epoch: The epoch we assume it is, needed by the lock
        - metadata: Data that will be attached to exceptions to help correlate errors
        - increase_epoch: whether the operation done in the context manager modifies things
            and thus the lock needs to increase the epoch to force other clients to
            update their data.
        """

        self.project_id: ProjectID = project_id
        self.lock: Lock = lock
        self.epoch: Epoch = epoch
        self.metadata: Any = metadata
        self.increase_epoch: bool = increase_epoch
        self._renewal_task: asyncio.Task | None = None
        self._stop_renewal: bool = False
        self._renewal_delay: float | None = None

    async def _renew_lock_task(self):
        """Background task to periodically renew the lock"""
        assert self._renewal_delay is not None, "_renew_lock_task called before __aenter__"
        while not self._stop_renewal:
            try:
                await asyncio.sleep(self._renewal_delay)
                # we might be cancelled during the sleep
                if self._stop_renewal:
                    break
                self._renewal_delay = await self.lock.renew_lock(self.project_id)
            except asyncio.CancelledError:
                break
            except LockError as e:
                logger.error(f"Failed to renew lock for project {self.project_id}: {e}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error renewing lock for project {self.project_id}: {e}")
                break

    async def __aenter__(self):
        """Acquire the lock and start a task that automatically renews the lock."""
        self._renewal_delay = await self.lock.lock(self.project_id, self.epoch, self.metadata)

        # Start the renewal task
        self._stop_renewal = False
        self._renewal_task = asyncio.create_task(self._renew_lock_task())

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Release the lock and cancel the automatic lock renewal task."""
        # Stop the renewal task
        self._stop_renewal = True
        if self._renewal_task:
            self._renewal_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._renewal_task

        # If an exception occurred, we don't increase the epoch
        increase_epoch = self.increase_epoch
        if exc_type is not None:
            increase_epoch = False

        # Unlock the project and get the new epoch
        try:
            self.epoch = await self.lock.unlock(self.project_id, increase_epoch=increase_epoch)
        except Exception as e:
            logger.error(f"Error unlocking project {self.project_id}: {e}")
            # Re-raise the exception if it's more critical than the original
            if exc_type is None:
                raise
