#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Optional

import jsonschema
import yaml

from .analysis import Analysis, AnalysisList
from .container import Container, ContainerDeclaration
from .pipeline import AnalysisBinding, Artifact, ArtifactCategory, Pipeline
from .pipeline_node import DummyPipelineNode, PipelineNode
from .task.pipe import Pipe
from .task.savepoint import SavePoint
from .utils.registry import get_registry


@dataclass(slots=True)
class Node:
    """
    We use this to represent the graph of branches in the pipeline,
    so we can compute the loading order of the branches.
    """

    content: Any
    is_root: bool = True
    successors: set[str] = field(default_factory=set)


def schema() -> dict[str, Any]:
    """
    Return the jsonschema for the pipeline.
    """
    root = Path(__file__).resolve().parent
    with open(root / "schema.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PipelineParser:
    def __init__(self, values: Any):
        # This setup is needed so we can have multiple schemas in the same file
        # so we can reuse the definitions
        full_schema = schema()
        resolver = jsonschema.RefResolver.from_schema(full_schema)
        pipeline_schema = full_schema["$defs"]["pipeline"]
        validator = jsonschema.Draft7Validator(pipeline_schema, resolver=resolver)
        validator.validate(values)

        self.values = values
        self.container_decls: dict[str, ContainerDeclaration] = {}
        self.artifacts: set[Artifact] = set()
        self.analyses: set[AnalysisBinding] = set()
        self.artifact_categories: dict[str, ArtifactCategory] = {}

    def parse(self) -> Pipeline:
        # Parse all the container declarations
        self.container_decls = self._parse_container_decls(self.values["containers"])

        # Parse all the artifact categories
        self.artifact_categories = self._parse_artifact_categories(
            self.values["artifact-categories"]
        )
        # Check that there is at least one artifact category
        assert len(self.artifact_categories) > 0

        # Then parse analyses lists, we will just do structural parsing, the actual validation
        # will be done in the Pipeline's __init__
        analysis_lists: list[AnalysisList] = []
        for analysis_list in self.values.get("analysis-lists", []):
            analysis_lists.append(
                AnalysisList(
                    name=analysis_list["name"],
                    analyses=analysis_list["analyses"],
                    description=analysis_list.get("description"),
                )
            )

        roots: list[PipelineNode] = self._parse_branches(branches=self.values["branches"])

        root_analyses_raw = self.values.get("analyses", [])
        if len(roots) == 1 and len(root_analyses_raw) == 0:
            root = roots[0]
        else:
            root = DummyPipelineNode()
            for node in roots:
                root.add_successor(node)
            root_analyses_dict = [{"analysis": x, "containers": []} for x in root_analyses_raw]
            self._parse_analyses(root_analyses_dict, root)

        # Check artifact consistency
        artifact_names = {a.name for a in self.artifacts}
        for artifact in self.artifacts:
            # TODO: also check for locations once those are statically defined
            for preferred_artifact in artifact.preferred_artifacts:
                if preferred_artifact not in artifact_names:
                    raise ValueError(
                        f"Artifact {artifact.name} defines {preferred_artifact} "
                        "as a preferred artifact but it does not exist"
                    )

        return Pipeline(
            declarations=set(self.container_decls.values()),
            root=root,
            artifacts=self.artifacts,
            analyses=self.analyses,
            analysis_lists=analysis_lists,
        )

    def _parse_pipe(self, task: dict[str, Any]):
        """Parse a pipe task from the JSON value."""
        pipes = get_registry(Pipe)  # type: ignore[type-abstract]
        pipe_name = task["pipe"]
        if pipe_name not in pipes:
            raise ValueError(
                f"Pipe {pipe_name} is not registered, available pipes: " f"{sorted(pipes.keys())}"
            )
        pipe_type = pipes[pipe_name]

        pipe_args = task.get("arguments", [])
        bindings = []
        for arg in pipe_args:
            if arg not in self.container_decls:
                raise ValueError(
                    f'While parsing {pipe_name}\'s arguments found container "{arg}" '
                    "that is not declared in the pipeline"
                )
            bindings.append(self.container_decls[arg])

        configuration = task.get("configuration")
        if configuration is None:
            configuration_string = ""
        elif isinstance(configuration, str):
            configuration_string = configuration
        else:
            configuration_string = yaml.safe_dump(configuration)

        return PipelineNode(
            task=pipe_type(configuration_string),
            bindings=bindings,
        )

    def _parse_savepoint(self, task: dict[str, Any]):
        """Parse a savepoint task from the JSON value."""
        name = task["savepoint"]
        containers = task["containers"]
        args = []
        for container_name in containers:
            if container_name not in self.container_decls:
                raise ValueError(f"Container {container_name} is not declared in the pipeline")
            args.append(self.container_decls[container_name])
        return PipelineNode(SavePoint(name=name, to_save=args))

    def _parse_artifacts(self, node_artifacts: list[Any], target_node: PipelineNode):
        """
        Parse artifacts from the node artifacts list and populate the artifacts dictionary.
        """
        for artifact in node_artifacts:
            name = artifact["name"]
            container = artifact["container"]
            if container not in self.container_decls:
                raise ValueError(
                    f"Artifact {name} references container {container} that is not "
                    "declared in the pipeline"
                )

            # Artifacts need to have a category associated to them
            if artifact["category"] not in self.artifact_categories:
                raise ValueError(f"Artifact category {artifact["category"]} not found")

            self.artifacts.add(
                Artifact(
                    name=name,
                    node=target_node,
                    container=self.container_decls[container],
                    description=artifact.get("description"),
                    category=self.artifact_categories[artifact["category"]],
                    filename=artifact.get("filename"),
                    defined_locations=artifact.get("defined_locations", []),
                    preferred_artifacts=artifact.get("preferred_artifacts", []),
                )
            )

    def _parse_analyses(self, node_analyses: list[Any], target_node: PipelineNode):
        """
        Parse analyses from the node analyses list and populate the analyses dictionary.
        """
        for analysis in node_analyses:
            analysis_name = analysis["analysis"]
            if analysis_name in self.analyses:
                raise ValueError(
                    f"Analysis {analysis_name} is defined multiple times in the pipeline"
                )
            containers = analysis["containers"]
            bindings = []
            for container in containers:
                if container not in self.container_decls:
                    raise ValueError(
                        f"Analysis {analysis_name} references container {container} that is "
                        "not declared in the pipeline"
                    )
                bindings.append(self.container_decls[container])

            analyses_registry = get_registry(Analysis)  # type: ignore[type-abstract]
            if analysis_name not in analyses_registry:
                raise ValueError(
                    f"Analysis {analysis_name} is not registered, available analyses: "
                    f"{sorted(analyses_registry.keys())}"
                )
            analysis_type: type[Analysis] = analyses_registry[analysis_name]
            self.analyses.add(
                AnalysisBinding(
                    analysis=analysis_type(),
                    bindings=tuple(bindings),
                    node=target_node,
                )
            )

    def _parse_task(self, task: Any, parent: Optional[PipelineNode] = None) -> PipelineNode:
        """
        Parse a single task from the JSON value.
        The JSON value should contain a dictionary with the branch content.
        """

        # Create the PipelineNode
        res: PipelineNode
        if "savepoint" in task:
            res = self._parse_savepoint(task)
        else:
            res = self._parse_pipe(task)

        # Parse artifacts
        node_artifacts = task.get("artifacts", [])
        self._parse_artifacts(node_artifacts=node_artifacts, target_node=res)

        # Parse analyses
        node_analyses = task.get("analyses", [])
        self._parse_analyses(node_analyses=node_analyses, target_node=res)

        # Connect
        if parent is not None:
            parent.add_successor(res)
        return res

    def _parse_branch(
        self, node: Node, graph: dict[str, PipelineNode]
    ) -> tuple[PipelineNode, PipelineNode]:
        """
        Parse the branch and return the last node in the branch.
        """
        parent: Optional[PipelineNode] = None
        if "from" in node.content:
            parent_name = node.content["from"]
            if parent_name not in graph:
                raise ValueError(f"Branch {parent_name} is not defined in the pipeline")
            parent = graph[parent_name]

        # Parse the nodes
        tasks = node.content.get("tasks", [])
        head = None
        for task in tasks:
            if "pipe" not in task and "savepoint" not in task:
                raise ValueError("Task must have either a pipe or a savepoint")
            if "pipe" in task and "savepoint" in task:
                raise ValueError("Task cannot have both a pipe and a savepoint")

            # Parse the node
            res = self._parse_task(task=task, parent=parent)
            if head is None:
                head = res
            parent = res

        assert parent is not None, "A branch cannot be empty and not have a parent"
        assert head is not None, "A branch needs to have at least one task"
        return (head, parent)

    def _parse_branches(self, branches: dict[str, Any]) -> list[PipelineNode]:
        """
        Parse the branches from the JSON value.
        The JSON value should contain a dictionary of branches with their names and tasks.
        """

        # Find the root branch
        graph: dict[str, Node] = {}
        roots: set[str] = set()
        for name, branch in branches.items():
            node = Node(content=branch)
            graph[name] = node
            if "from" in branch:
                node.is_root = False
                from_node = graph[branch["from"]]
                from_node.successors.add(name)
            else:
                roots.add(name)

        sorter: TopologicalSorter[str] = TopologicalSorter()
        for name, node in graph.items():
            sorter.add(name)
            for successor in node.successors:
                sorter.add(successor, name)

        # Parse the branches in the correct order (DFS)
        node_tips: dict[str, PipelineNode] = {}
        result: list[PipelineNode] = []
        for name in sorter.static_order():
            first_node, last_node = self._parse_branch(node=graph[name], graph=node_tips)
            node_tips[name] = last_node
            if name in roots:
                result.append(first_node)

        return result

    def _parse_container_decls(self, containers: list[Any]) -> dict[str, ContainerDeclaration]:
        """
        Parse the container declarations from the JSON value.
        The JSON value should contain a list of containers with their names and types.
        """
        containers_registry = get_registry(Container)  # type: ignore[type-abstract]
        container_decls: dict[str, ContainerDeclaration] = {}
        for container in containers:
            name = container["name"]
            ty = container["type"]
            if ty not in containers_registry:
                raise ValueError(
                    f"Container type {ty} is not registered, the available types "
                    f"are: {sorted(containers_registry.keys())}"
                )

            container_decls[name] = ContainerDeclaration(
                name=name,
                container_type=containers_registry[ty],
            )
        return container_decls

    def _parse_artifact_categories(
        self, artifact_categories: list[Any]
    ) -> dict[str, ArtifactCategory]:
        result: dict[str, ArtifactCategory] = {}
        for element in artifact_categories:
            if element["name"] in result:
                raise ValueError(f"Artifact category {element["name"]} is duplicated")
            result[element["name"]] = ArtifactCategory(element["name"], element["show-by-default"])
        return result


def load_pipeline_yaml(yaml_data: str) -> Pipeline:
    """
    Load a pipeline from a YAML string.
    """
    values = yaml.safe_load(yaml_data)
    return PipelineParser(values).parse()


def load_pipeline_yaml_file(file: str) -> Pipeline:
    """
    Load a pipeline from a YAML file."""
    with open(file, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)
    return PipelineParser(values).parse()
