#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Any, Optional

from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.object import Kind, ObjectID, ObjectSet
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.utils.registry import get_singleton


class DaemonException(Exception):
    """
    An exception that a route can raise to do an early-exit with
    some data and a status code.
    """

    def __init__(self, status_code: int, data: Any):
        self.code = status_code
        self.data = data


Epoch = int


def compute_objects(
    model: ReadOnlyModel,
    kind: Kind,
    objects: Optional[list[str]],
) -> ObjectSet:
    """
    If objects is not None, deserialize all objects, otherwise return all the objects producible of
    the given Kind.
    """
    # If the user did not provide any objects for this container,
    # we will use all objects of the given kind from the model.
    if objects is None or len(objects) == 0:
        return model.all_objects(kind)
    # Otherwise we have to parse the objects provided by the user.

    # Parse the objects into ObjectSet
    objset = set()
    for obj in objects:
        if not isinstance(obj, str):
            raise DaemonException(
                400,
                {
                    "msg": f'Object "{obj}" must be a string, got "{type(obj)}"',
                },
            )

        # Deserialize the object ID
        obj_type = get_singleton(ObjectID)  # type: ignore [type-abstract]
        try:
            obj_id = obj_type.deserialize(obj)
        except ValueError as e:
            raise DaemonException(
                400,
                {
                    "msg": f'Invalid object ID "{obj}": {e}',
                },
            )

        objset.add(obj_id)

    return ObjectSet(kind, objset)


def compute_artifact(
    storage_provider: StorageProvider,
    pipeline: Pipeline,
    model: ReadOnlyModel,
    artifact_name: str,
    artifact_data: dict,
) -> dict[str, Any]:
    """
    Compute the requested artifact and return it in a HTTP compatible format.
    """
    artifact_type = pipeline.artifacts[artifact_name]
    artifact = pipeline.artifacts[artifact_name]
    container = pipeline.get_artifact(
        model=model,
        artifact=artifact,
        requests=compute_objects(
            model=model,
            kind=artifact_type.container.container_type.kind,
            objects=artifact_data.get("objects"),
        ),
        pipeline_configuration=artifact_data.get("configuration", {}),
        storage_provider=storage_provider,
    )
    return {
        "cacheable": artifact.is_cacheable(),
        "objects": container.to_dict(),
    }
