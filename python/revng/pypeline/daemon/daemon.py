#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import base64
from dataclasses import dataclass, field
from typing import Any

from revng.pypeline.container import Container
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import Kind
from revng.pypeline.storage.storage_provider import storage_provider_factory_factory
from revng.pypeline.task.requests import Requests
from revng.pypeline.utils.registry import get_registry, get_singleton

from .lock import Lock
from .utils import compute_artifact, compute_objects


@dataclass
class Response:
    """A simple wrapper used to enforce correctness and a standard format for
    responses."""

    code: int
    """The HTTP status code of the response."""
    body: Any
    """
    The response body of the request.
    """
    headers: dict[str, str] = field(default_factory=dict)
    """The headers of the response."""
    notifications: list[Any] = field(default_factory=list)
    """List of notifications to send through the NotificationPublisher"""

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
        version: str,
        pipeline_yaml: Any,
        pipeline: Any,
        lock: Lock,
        debug: bool,
        storage_provider_url: str,
        cache_dir: str,
    ):
        self.version = version
        self.pipeline_yaml = pipeline_yaml
        self.pipeline = pipeline
        self.lock = lock
        self.debug = debug
        self.storage_provider_factory = storage_provider_factory_factory(storage_provider_url)
        self.cache_dir = cache_dir

    def get_epoch(self, request) -> Response:
        return Response(code=200, body={"epoch": self.lock.get_epoch(request["project_id"])})

    def get_model(self, request):
        project_id = request["project_id"]
        token = request.get("token")
        epoch = self.lock.get_epoch(project_id)

        storage_provider = self.storage_provider_factory.get(
            project_id=project_id,
            token=token,
            cache_dir=self.cache_dir,
        )

        model_ty: type[Model] = get_singleton(Model)
        model: bytes = storage_provider.get_model()

        # Optimize the model serialization
        if model_ty.is_text():
            model_str = model.decode("utf-8")
        else:
            model_str = base64.b64encode(model).decode("utf-8")

        return Response(
            code=200,
            body={
                "epoch": epoch,
                "model_type": model_ty.__name__,
                "mime_type": model_ty.mime_type(),
                "is_text": model_ty.is_text(),
                "model": model_str,
            },
        )

    def get_pipeline(self) -> Response:
        # This for could be a list comprehension, but mypy can't understand the
        # conditional serialization on parent
        kinds = []
        for kind in get_singleton(Kind).kinds():  # type: ignore [type-abstract]
            parent = kind.parent()
            if parent is not None:
                parent_name = parent.serialize()
            else:
                parent_name = None
            kinds.append(
                {
                    "name": kind.serialize(),
                    "parent": parent_name,
                }
            )

        return Response(
            code=200,
            body={
                "version": self.version,
                "pipeline": self.pipeline_yaml,
                "containers": [
                    {
                        "name": name,
                        "kind": container.kind.serialize(),
                        "mime_type": container.mime_type(),
                        "is_text": container.is_text(),
                    }
                    for name, container in get_registry(
                        Container  # type: ignore [type-abstract]
                    ).items()
                ],
                "artifacts": [
                    {
                        "name": artifact.name,
                        "container_name": artifact.container.name,
                        "container_type": artifact.container.container_type.__name__,
                        "cacheable": artifact.is_cacheable(),
                    }
                    for artifact in self.pipeline.artifacts.values()
                ],
                "kinds": kinds,
            },
        )

    def artifact(self, request) -> Response:
        model_ty: type[Model] = get_singleton(Model)  # type: ignore [type-abstract]
        project_id = request["project_id"]
        token = request.get("token")
        artifacts = request["artifacts"]

        # Validate data
        for artifact_name, _ in artifacts.items():
            if artifact_name not in self.pipeline.artifacts:
                return Response(
                    code=400,
                    body={
                        "msg": f"Artifact {artifact_name} not found in the pipeline.",
                        "available_artifacts": list(self.pipeline.artifacts.keys()),
                    },
                )

        # Create a storage provider
        storage_provider = self.storage_provider_factory.get(
            project_id=project_id,
            token=token,
            cache_dir=self.cache_dir,
        )
        # Load the model
        model = model_ty()
        model.deserialize(storage_provider.get_model())

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

    def analyze(self, request) -> Response:
        """Process analysis requests"""
        project_id = request["project_id"]

        model_ty: type[Model] = get_singleton(Model)  # type: ignore [type-abstract]

        # Extract the data
        epoch = request["epoch"]
        token = request.get("token")
        analysis = request["analysis"]
        configuration = request.get("configuration", "")
        pipeline_configuration = request.get("pipeline_configuration", {})
        containers = request.get("containers", {})

        # Validate data
        if analysis not in self.pipeline.analyses:
            return Response(
                code=400,
                body={
                    "msg": f"Analysis {analysis} not found in the pipeline.",
                    "available_analyses": list(self.pipeline.analyses.keys()),
                },
            )

        for container_name, objects in containers.items():
            for decl in self.pipeline.declarations:
                if container_name == decl.name:
                    break
            else:
                return Response(
                    code=400,
                    body={
                        "msg": f"Container {container_name} not found in the pipeline.",
                        "available_containers": sorted(
                            decl.name for decl in self.pipeline.declarations
                        ),
                    },
                )

        # Create a storage provider
        storage_provider = self.storage_provider_factory.get(
            project_id=project_id,
            token=token,
            cache_dir=self.cache_dir,
        )
        # Load the model
        model = model_ty()
        model.deserialize(storage_provider.get_model())

        # Setup the requests
        requests = Requests()
        for binding in self.pipeline.analyses[analysis].bindings:
            kind: Kind = binding.container_type.kind
            objects = containers.get(binding.name)
            if objects is not None and not isinstance(objects, list):
                return Response(
                    code=400,
                    body={
                        "msg": (
                            f"Objects for container {binding.name} must be a "
                            f"list, got {type(objects)}",
                        ),
                    },
                )
            requests.insert(binding, compute_objects(ReadOnlyModel(model), kind, objects))

        # Run the analysis
        new_model = self.pipeline.run_analysis(
            model=ReadOnlyModel(model),
            analysis_name=analysis,
            requests=requests,
            analysis_configuration=configuration,
            pipeline_configuration=pipeline_configuration,
            storage_provider=storage_provider,
        )

        diff = str(model.diff(new_model))

        # Decide whether to increment the epoch
        new_epoch = epoch
        if len(diff) != 0:
            new_epoch = epoch + 1

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
                    "invalidated": [
                        # WIP List of invalidated cacheable artifacts (and the objects inside it)
                    ],
                }
            ],
        )
