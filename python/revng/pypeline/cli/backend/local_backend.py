#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from revng.pypeline.container import Container
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.pipeline import AnalysisList, Artifact, Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.storage_provider import FileStorageEntry, InvalidatedObjects, LockType
from revng.pypeline.storage.storage_provider import ProjectID, StorageProvider
from revng.pypeline.storage.storage_provider import StorageProviderFactory
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils.registry import get_registry

from .backend import Backend, BackendFactory, BackendFeature


class LocalBackend(Backend):
    """An in-process backend: forwards to the `Pipeline`, computing locally."""

    def __init__(
        self,
        pipeline: Pipeline,
        storage_provider: StorageProvider,
        runner_context: RunnerContext,
    ):
        self._pipeline = pipeline
        self._storage_provider = storage_provider
        self._runner_context = runner_context

    def get_model(self, configuration: PipelineConfiguration) -> tuple[Model, int]:
        return self._pipeline.get_model(configuration, self._storage_provider)

    def get_artifact(
        self,
        model: ReadOnlyModel,
        artifact: Artifact,
        requests: ObjectSet,
        configuration: PipelineConfiguration,
    ) -> Container:
        return self._pipeline.get_artifact(
            model=model,
            artifact=artifact,
            requests=requests,
            configuration=configuration,
            storage_provider=self._storage_provider,
            runner_context=self._runner_context,
        )

    def run_analysis(
        self,
        model: ReadOnlyModel,
        analysis_name: str,
        requests: Requests,
        configuration: PipelineConfiguration,
    ) -> tuple[Model, InvalidatedObjects]:
        return self._pipeline.run_analysis(
            model=model,
            analysis_name=analysis_name,
            requests=requests,
            configuration=configuration,
            storage_provider=self._storage_provider,
            runner_context=self._runner_context,
        )

    def run_analysis_list(
        self,
        model: ReadOnlyModel,
        analysis_list: AnalysisList,
        configuration: PipelineConfiguration,
    ) -> tuple[Model, InvalidatedObjects]:
        return self._pipeline.run_analysis_list(
            model=model,
            analysis_list=analysis_list,
            configuration=configuration,
            storage_provider=self._storage_provider,
            runner_context=self._runner_context,
        )

    def put_files(self, files: list[FileStorageEntry]) -> list[str]:
        return self._storage_provider.put_files_in_storage(files)


class LocalBackendFactory(BackendFactory):
    """Produces backends that compute in-process, persisting through the storage
    provider at `url`."""

    # In-process compute can inspect debug subprocesses, report invalidations
    # and initialize a new project.
    features = BackendFeature.INIT | BackendFeature.DEBUG | BackendFeature.INVALIDATIONS

    @classmethod
    def schemes(cls) -> list[str]:
        # The local backend handles every registered storage-provider scheme.
        registry = get_registry(StorageProviderFactory)  # type: ignore[type-abstract]
        return [factory.scheme() for factory in registry.values()]

    def __init__(self, url: str, *, pipeline: Pipeline, base_directory: Path, cache_dir: str):
        self._factory = storage_provider_factory_factory(url)
        self._pipeline = pipeline
        self._base_directory = base_directory
        self._cache_dir = cache_dir

    @asynccontextmanager
    async def session(
        self,
        *,
        lock_type: LockType,
        project_id: ProjectID | None,
        token: str | None,
        runner_context: RunnerContext = RunnerContext(),
    ) -> AsyncIterator[Backend]:
        storage_provider_context = self._factory.get(
            base_directory=self._base_directory,
            pipeline=self._pipeline,
            lock_type=lock_type,
            project_id=project_id,
            token=token,
            cache_dir=self._cache_dir,
        )
        async with storage_provider_context as storage_provider:
            yield LocalBackend(self._pipeline, storage_provider, runner_context)
