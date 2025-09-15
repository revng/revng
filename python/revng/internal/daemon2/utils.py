#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import base64
from typing import Optional, Any

from revng.pypeline.pipeline import Pipeline
from revng.pypeline.object import ObjectID, ObjectSet, Kind
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.utils.registry import get_singleton

class BasicException(Exception):
    """
    An exception that a route can raise to do an early-exit with
    some data and a status code.
    """
    def __init__(self, status_code: int, data: Any):
        self.code = status_code
        self.data = data

ProjectID = str
Epoch = int

def compute_objects(
    model: Model,
    kind: Kind,
    objects: Optional[list[str]],
) -> ObjectSet:
    """
    If objects is not None, deserialize all objects, otherwise return all the objects producible of
    the given Kind.
    """
    # if the user did not provide any objects for this container,
    # we will use all objects of the given kind from the model.
    if objects is None or len(objects) == 0:
        return model.all_objects(kind)
    # Otherwise we have to parse the objects provided by the user.

    # Parse the objects into ObjectSet
    objset = set()
    for obj in objects:
        if not isinstance(obj, str):
            raise BasicException(
                400,
                {
                    "msg": f"Object `{obj}` must be a string, got `{type(obj)}`",
                }
            )

        # Deserialize the object ID
        obj_ty = get_singleton(ObjectID)
        try:
            obj_id = obj_ty.deserialize(obj)
        except ValueError as e:
            raise BasicException(
                400,
                {
                    "msg": f"Invalid object ID `{obj}`: {e}",
                }
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
    artifact_ty = pipeline.artifacts[artifact_name]
    container = pipeline.get_artifact(
        model=model,
        artifact=pipeline.artifacts[artifact_name],
        requests=compute_objects(
            model=model,
            kind=artifact_ty.container.container_type.kind,
            objects=artifact_data.get("objects", None),
        ),
        pipeline_configuration=artifact_data.get("configuration", {}),
        storage_provider=storage_provider,
    )
    # Return the objects with their serialized data
    objects = {
        obj_id.serialize(): (
            base64.b64encode(data).decode("utf-8") if container.is_text() else data.decode("utf-8")
        )
        for obj_id, data in container.serialize().items()
    }
    return {
        "cachable": False,  # TODO: how?
        "objects": objects,
    }
