#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path
from typing import Any

import jsonschema
import yaml

import revng.pypeline
from revng.pypeline.container import Container
from revng.pypeline.model import Model
from revng.pypeline.object import Kind
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.utils.registry import get_registry, get_singleton


def get_pipeline_description(pipeline: Pipeline) -> dict[str, Any]:
    """
    Build and validate the web representation of the pipeline.
    """

    model_type: type[Model] = get_singleton(Model)  # type: ignore [type-abstract]
    # Build the pipeline description
    pipeline_description = {
        "version": revng.pypeline.__version__,
        "model": {
            "identifier": model_type.identifier,
            "name": model_type.model_name(),
            "mime_type": model_type.mime_type(),
            "is_text": model_type.is_text(),
        },
        "kinds": get_singleton(Kind).type_dict(),  # type: ignore
        "container_types": [
            container.type_dict()
            for container in get_registry(Container).values()  # type: ignore [type-abstract]
        ],
        "containers": {
            declaration.name: declaration.container_type.name
            for declaration in pipeline.declarations
        },
        "root_node_id": pipeline.root.id,
        "nodes": [node.to_dict() for node in pipeline.walk_pipeline(stable=True)],
        "artifacts": [artifact.to_dict() for artifact in pipeline.artifacts.values()],
        "analyses": [analysis.to_dict() for analysis in pipeline.analyses.values()],
        "analyses_lists": [
            analysis_list.to_dict() for analysis_list in pipeline.analysis_lists.values()
        ],
    }

    # Ensure that it respects the agreed schema
    root = Path(__file__).resolve().parent.parent
    with open(root / "pipeline-description-schema.yml", "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    validator = jsonschema.Draft7Validator(schema)
    validator.validate(pipeline_description)

    return pipeline_description
