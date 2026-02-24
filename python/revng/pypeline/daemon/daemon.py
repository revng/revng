#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import Kind
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.storage.storage_provider import FileStorageEntry
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils import bytes_to_string
from revng.pypeline.utils.pipeline import get_pipeline_description
from revng.pypeline.utils.registry import get_singleton

from .exceptions import EpochError, MalformedRequestError
from .utils import compute_artifact, compute_objects


@dataclass
class Response:
    """A simple wrapper used to enforce correctness and a standard format for
    responses."""

    code: int
    """The HTTP status code of the response."""
    body: Any
    """The response body of the request."""
    headers: dict[str, str] = field(default_factory=dict)
    """The headers of the response."""
    notifications: list[Any] = field(default_factory=list)
    """List of notifications to send through the NotificationBroker."""

    def to_dict(self):
        result = {
            "code": self.code,
            "body": self.body,
        }
        if len(self.headers) != 0:
            result["headers"] = self.headers
        if len(self.notifications) != 0:
            result["notifications"] = self.notifications
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

    def _get_storage_provider_context(self, request):
        project_id = request.get("project_id")
        token = request.get("token")
        return self.storage_provider_factory.get(
            base_directory=self.base_directory,
            project_id=project_id,
            token=token,
            cache_dir=self.cache_dir,
        )

    async def get_epoch(self, request) -> Response:
        storage_provider_context = self._get_storage_provider_context(request)
        async with storage_provider_context as storage_provider:
            return Response(code=200, body={"epoch": storage_provider.get_epoch()})

    async def get_model(self, request):
        storage_provider_context = self._get_storage_provider_context(request)
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
        storage_provider_context = self._get_storage_provider_context(request)
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
        artifacts = request["artifacts"]
        epoch = request["epoch"]

        # Validate data
        for artifact_name, _ in artifacts.items():
            if artifact_name not in self.pipeline.artifacts:
                raise MalformedRequestError(f"Artifact {artifact_name} not found in the pipeline")

        storage_provider_context = self._get_storage_provider_context(request)
        async with storage_provider_context as storage_provider:
            # Load the model
            model, real_epoch = storage_provider.get_model()

            if real_epoch != epoch:
                raise EpochError(real_epoch, epoch)

            # Process each artifact
            res = {}
            for artifact_name, artifact_data in artifacts.items():
                # Compute the artifact
                res[artifact_name] = compute_artifact(
                    storage_provider=storage_provider,
                    pipeline=self.pipeline,
                    model=ReadOnlyModel(model),
                    artifact_name=artifact_name,
                    artifact_data=artifact_data,
                )

        # Return the artifacts
        return Response(code=200, body={"artifacts": res})

    async def analyze(self, request) -> Response:
        """Process analysis requests"""

        # Extract the data
        epoch = request["epoch"]
        analysis = request["analysis"]
        configuration = request.get("configuration", "")
        pipeline_configuration = request.get("pipeline_configuration", {})
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

        storage_provider_context = self._get_storage_provider_context(request)
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
                    analysis_configuration=configuration,
                    pipeline_configuration=pipeline_configuration,
                    storage_provider=storage_provider,
                )
            else:
                analysis_list = self.pipeline.analysis_lists[analysis]
                if len(configuration) == 0:
                    configuration = ["" for _ in analysis_list.analyses]
                new_model, invalidated = self.pipeline.run_analysis_list(
                    model=ReadOnlyModel(model),
                    analysis_list=analysis_list,
                    analysis_configuration=configuration,
                    pipeline_configuration=pipeline_configuration,
                    storage_provider=storage_provider,
                )

            # Compute the diff between the original model and the final one
            diff_raw = model.diff(new_model).serialize()
            # TODO: this can be done much more efficiently
            new_epoch = storage_provider.get_epoch()

        # Only return cacheable artifacts invalidations
        invalidated_artifacts: list[dict[str, Any]] = []
        for container_location, object_ids in invalidated.items():
            artifact = self.pipeline.savepoint_id_to_artifact.get(container_location.savepoint_id)
            if artifact is None:
                continue
            invalidated_artifacts.append(
                {
                    "name": artifact.name,
                    "configuration": container_location.configuration_id,
                    "object_ids": [object_id.serialize() for object_id in object_ids],
                }
            )

        model_type = get_singleton(Model)  # type: ignore[type-abstract]
        diff = bytes_to_string(diff_raw, model_type.is_text())
        # Return the updated model
        return Response(
            code=200,
            body={
                "epoch": new_epoch,
                "diff": diff,
            },
            notifications=[
                {
                    "type": "analysis",
                    "analysis": analysis,
                    "epoch": new_epoch,
                    "diff": diff,
                    "invalidated": invalidated_artifacts,
                }
            ],
        )
