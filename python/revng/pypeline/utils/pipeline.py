#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path
from typing import Any

import jsonschema
import yaml

import revng.pypeline
from revng.pypeline.analysis import Analysis
from revng.pypeline.container import Container
from revng.pypeline.model import Model
from revng.pypeline.object import Kind
from revng.pypeline.pipeline import Pipeline
from revng.pypeline.pipeline_node import PipelineConfiguration
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.utils import PypelineException
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


def deserialize_configuration(pipeline: Pipeline, input_: dict[str, str]) -> PipelineConfiguration:
    """Given a `dict[str, str]`, map it to a PipelineConfiguration by walking
    the pipeline tree and assigning keys based on the pipe/analysis name
    """

    pipes = get_registry(Pipe)  # type: ignore[type-abstract]
    analyses = get_registry(Analysis)  # type: ignore[type-abstract]
    for key in input_:
        if key not in pipes and key not in analyses:
            raise PypelineException(f"Passed configuration key '{key}' does not exist")

    configuration: dict[Pipe | Analysis, str] = {}
    for node in pipeline.walk_pipeline():
        if not isinstance(node.task, Pipe):
            continue
        if node.task.name in input_:
            configuration[node.task] = input_[node.task.name]

    for binding in pipeline.analyses.values():
        if binding.analysis.name in input_:
            configuration[binding.analysis] = input_[binding.analysis.name]

    return configuration
