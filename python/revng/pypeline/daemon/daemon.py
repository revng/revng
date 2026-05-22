#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from revng.pypeline.container import ContainerFormat
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import Kind
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.storage.storage_provider import FileStorageEntry, LockType
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils import bytes_to_string
from revng.pypeline.utils.pipeline import deserialize_configuration, get_pipeline_description
from revng.pypeline.utils.registry import get_singleton

from .exceptions import EpochError, MalformedRequestError
from .utils import compute_objects


@dataclass
class Response:
    """A simple wrapper used to enforce correctness and a standard format for
    responses."""

    code: int
    """The HTTP status code of the response."""
    body: Any
    """The response body of the request."""
    content_type: str | None = None
    """The MIME of the response content."""
    headers: dict[str, str] = field(default_factory=dict)
    """The headers of the response."""

    def to_dict(self):
        result = {
            "code": self.code,
            "body": self.body,
        }
        if len(self.headers) != 0:
            result["headers"] = self.headers
        return result


class Daemon:
    """The transport agnostic part of the daemon."""

    def __init__(
        self,
        pipeline: Pipeline,
        storage_provider_url: str,
        cache_dir: str,
        base_directory: Path,
    ):
        self.pipeline = pipeline
        self.cache_dir = cache_dir
        self.base_directory = base_directory
        self.storage_provider_factory = storage_provider_factory_factory(storage_provider_url)
        self.pipeline_description = get_pipeline_description(pipeline)

    def _get_storage_provider_context(self, request, lock_type: LockType):
        project_id = request.get("project_id")
        token = request.get("token")
        return self.storage_provider_factory.get(
            base_directory=self.base_directory,
            pipeline=self.pipeline,
            lock_type=lock_type,
            project_id=project_id,
            token=token,
            cache_dir=self.cache_dir,
        )

    async def get_epoch(self, request) -> Response:
        storage_provider_context = self._get_storage_provider_context(request, LockType.ARTIFACT)
        async with storage_provider_context as storage_provider:
            return Response(code=200, body={"epoch": storage_provider.get_epoch()})

    async def get_model(self, request):
        storage_provider_context = self._get_storage_provider_context(request, LockType.ARTIFACT)
        async with storage_provider_context as storage_provider:
            model, epoch = storage_provider.get_model()

        return Response(
            code=200,
            body={
                "epoch": epoch,
                "model": bytes_to_string(model.serialize(), is_text=model.__class__.is_text()),
            },
        )

    def get_pipeline(self) -> Response:
        return Response(
            code=200,
            body=self.pipeline_description,
        )

    async def put_file(self, request) -> Response:
        entry = FileStorageEntry(request["name"], contents=request["contents"])
        storage_provider_context = self._get_storage_provider_context(request, LockType.ARTIFACT)
        async with storage_provider_context as storage_provider:
            hashes = storage_provider.put_files_in_storage([entry])

        return Response(
            code=200,
            body={
                "name": request["name"],
                "hash": hashes[0],
            },
        )

    async def artifact(self, request) -> Response:
        artifact_name: str = request["artifact"]
        objects: list[str] | None = request.get("objects")
        raw_configuration: dict[str, str] = request.get("configuration", {})
        epoch: int = request["epoch"]
        format_: str = request.get("format", "json")

        # Validate data
        if artifact_name not in self.pipeline.artifacts:
            raise MalformedRequestError(f"Artifact {artifact_name} not found in the pipeline")

        if format_ not in ("json", "tar"):
            raise MalformedRequestError(f"Format {format_} is not valid, valid values: json, tar")

        # Convert configuration
        configuration = deserialize_configuration(self.pipeline, raw_configuration)
        artifact = self.pipeline.artifacts[artifact_name]
        configuration_hash = artifact.node.configuration_id(configuration)

        # Compute the artifact
        storage_provider_context = self._get_storage_provider_context(request, LockType.ARTIFACT)
        async with storage_provider_context as storage_provider:
            # Load the model
            model, real_epoch = storage_provider.get_model()

            if real_epoch != epoch:
                raise EpochError(real_epoch, epoch)

            object_set = compute_objects(
                model=ReadOnlyModel(model),
                kind=artifact.container.container_type.kind,
                objects=objects,
            )

            container = self.pipeline.get_artifact(
                model=ReadOnlyModel(model),
                artifact=artifact,
                requests=object_set,
                configuration=configuration,
                storage_provider=storage_provider,
            )

        headers = {"x-pypeline-configuration-hash": configuration_hash}
        if format_ == "json":
            return Response(
                code=200,
                body={
                    key: bytes_to_string(value, container.is_text())
                    for key, value in container.to_dict(object_set).items()
                },
                headers=headers,
            )
        else:
            return Response(
                code=200,
                body=container.to_bytes(object_set, ContainerFormat.TAR),
                content_type="application/x-tar",
                headers=headers,
            )

    async def analyze(self, request) -> Response:
        """Process analysis requests"""

        # Extract the data
        epoch = request["epoch"]
        analysis = request["analysis"]
        raw_configuration = request.get("configuration", {})
        containers = request.get("containers", {})

        # Validate data and normalize to analysis list
        if analysis not in self.pipeline.analyses and analysis not in self.pipeline.analysis_lists:
            raise MalformedRequestError(f"Analysis {analysis} not found in the pipeline")

        # Check that the given containers are declared in the pipeline
        for container_name, objects in containers.items():
            for decl in self.pipeline.declarations:
                if container_name == decl.name:
                    break
            else:
                raise MalformedRequestError(f"Container {container_name} not found in the pipeline")

        configuration = deserialize_configuration(self.pipeline, raw_configuration)
        storage_provider_context = self._get_storage_provider_context(request, LockType.ANALYSIS)
        async with storage_provider_context as storage_provider:
            # Load the model
            model, real_epoch = storage_provider.get_model()

            if real_epoch != epoch:
                raise EpochError(real_epoch, epoch)

            # Run an analysis
            if analysis in self.pipeline.analyses:
                current_model = ReadOnlyModel(model)
                # Setup the requests
                requests = Requests()
                for binding in self.pipeline.analyses[analysis].bindings:
                    kind: Kind = binding.container_type.kind
                    objects = containers.get(binding.name)
                    if objects is not None and not isinstance(objects, list):
                        raise MalformedRequestError(
                            f"Objects for container {binding.name} must be a list, "
                            f"got {type(objects)}"
                        )
                    requests.insert(binding, compute_objects(current_model, kind, objects))

                # Run the analysis
                new_model, invalidated = self.pipeline.run_analysis(
                    model=current_model,
                    analysis_name=analysis,
                    requests=requests,
                    configuration=configuration,
                    storage_provider=storage_provider,
                )
            else:
                analysis_list = self.pipeline.analysis_lists[analysis]
                new_model, invalidated = self.pipeline.run_analysis_list(
                    model=ReadOnlyModel(model),
                    analysis_list=analysis_list,
                    configuration=configuration,
                    storage_provider=storage_provider,
                )

            # Compute the diff between the original model and the final one
            diff_raw = model.diff(new_model).serialize()
            # TODO: this can be done much more efficiently
            new_epoch = storage_provider.get_epoch()

        model_type = get_singleton(Model)  # type: ignore[type-abstract]
        diff = bytes_to_string(diff_raw, model_type.is_text())
        # Return the updated model
        return Response(code=200, body={"epoch": new_epoch, "diff": diff})
