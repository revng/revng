#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
The backend abstraction decouples the `project` CLI commands from *where* the
compute happens. A command resolves its arguments against the model and formats
the output; the backend performs the model load, the artifact production and the
analysis runs. `LocalBackendFactory` produces backends that compute in-process
(the historical behavior), `DaemonBackendFactory` produces backends that
delegate to a running daemon over HTTP.
"""

from abc import ABC, abstractmethod
from enum import Flag, auto
from pathlib import Path
from typing import AsyncContextManager, ClassVar
from urllib.parse import urlparse

from revng.pypeline.container import Container
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.pipeline import AnalysisList, Artifact, Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.storage_provider import FileStorageEntry, InvalidatedObjects, LockType
from revng.pypeline.storage.storage_provider import ProjectID
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils.registry import get_registry


class BackendFeature(Flag):
    """Optional capabilities a backend may or may not provide. A command tests
    membership (e.g. `BackendFeature.DEBUG in backend_factory.features`) before
    relying on the corresponding behavior."""

    # `project init` can create a new project from scratch. A daemon manages its
    # own project, so a daemon-backed factory does not offer it.
    INIT = auto()
    # `--debug` can run pipes and analyses as inspectable local subprocesses with
    # on-disk intermediate files. Backends that compute elsewhere cannot.
    DEBUG = auto()
    # `--invalidations` can report the objects an analysis invalidated. A remote
    # backend that only returns a model diff cannot.
    INVALIDATIONS = auto()


class Backend(ABC):
    """
    A single unit of work against a backend. Mirrors the subset of `Pipeline`
    that the `project` commands need, minus the storage provider and runner
    context (which the backend owns). The `model` passed to the compute methods
    is the one returned by `get_model`; the local backend uses it, the daemon
    backend relies on it being consistent with the epoch it holds.
    """

    @abstractmethod
    def get_model(self, configuration: PipelineConfiguration) -> tuple[Model, int]:
        """Load the model and its epoch."""

    @abstractmethod
    def get_artifact(
        self,
        model: ReadOnlyModel,
        artifact: Artifact,
        requests: ObjectSet,
        configuration: PipelineConfiguration,
    ) -> Container:
        """Produce the given artifact for the requested objects."""

    @abstractmethod
    def run_analysis(
        self,
        model: ReadOnlyModel,
        analysis_name: str,
        requests: Requests,
        configuration: PipelineConfiguration,
    ) -> tuple[Model, InvalidatedObjects]:
        """Run a single analysis and return the new model and invalidations."""

    @abstractmethod
    def run_analysis_list(
        self,
        model: ReadOnlyModel,
        analysis_list: AnalysisList,
        configuration: PipelineConfiguration,
    ) -> tuple[Model, InvalidatedObjects]:
        """Run an analysis list and return the new model and invalidations."""

    @abstractmethod
    def put_files(self, files: list[FileStorageEntry]) -> list[str]:
        """Store input files, returning their hashes."""


class BackendFactory(ABC):
    """A producer that hands out `Backend`s."""

    # The capabilities the backends this factory produces provide; see
    # `BackendFeature`.
    features: ClassVar[BackendFeature] = BackendFeature(0)

    @abstractmethod
    def __init__(self, url: str, *, pipeline: Pipeline, base_directory: Path, cache_dir: str):
        """
        Build the factory for the given backend URL. All factories share this
        signature so `backend_factory_for` can construct any of them uniformly;
        `base_directory` and `cache_dir` are only meaningful for backends that
        keep data locally and are ignored by the others.
        """

    @classmethod
    @abstractmethod
    def schemes(cls) -> list[str]:
        """The URL schemes that select this factory."""

    @abstractmethod
    def session(
        self,
        *,
        lock_type: LockType,
        project_id: ProjectID | None,
        token: str | None,
        runner_context: RunnerContext,
    ) -> AsyncContextManager[Backend]:
        """
        Open a backend. The context manager mirrors the storage provider's: it
        may take a lock for the duration of the work and release it on exit.
        """


def backend_factory_for(
    url: str,
    *,
    pipeline: Pipeline,
    base_directory: Path,
    cache_dir: str,
) -> BackendFactory:
    """
    Build the factory that handles the given backend URL, looked up by scheme in
    the factory registry: a `daemon://` URL selects the daemon factory, a
    storage-provider URL selects the local factory.
    """
    scheme = urlparse(url).scheme
    for factory_type in get_registry(BackendFactory).values():  # type: ignore[type-abstract]
        if scheme in factory_type.schemes():
            return factory_type(
                url, pipeline=pipeline, base_directory=base_directory, cache_dir=cache_dir
            )
    raise ValueError(f'No backend handles the URL scheme "{scheme}"')
