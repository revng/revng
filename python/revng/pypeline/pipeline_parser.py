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

from .analysis import Analysis, AnalysisBinding
from .container import Container, ContainerDeclaration
from .pipeline import Artifact, Pipeline
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


def parse_pipe(
    task: dict[str, Any],
    container_decls: dict[str, ContainerDeclaration],
):
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
        if arg not in container_decls:
            raise ValueError(
                f'While parsing {pipe_name}\'s arguments found container "{arg}" '
                "that is not declared in the pipeline"
            )
        bindings.append(container_decls[arg])

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


def parse_savepoint(
    task: dict[str, Any],
    container_decls: dict[str, ContainerDeclaration],
):
    """Parse a savepoint task from the JSON value."""
    name = task["savepoint"]
    containers = task["containers"]
    args = []
    for container_name in containers:
        if container_name not in container_decls:
            raise ValueError(f"Container {container_name} is not declared in the pipeline")
        args.append(container_decls[container_name])
    return PipelineNode(SavePoint(name=name, to_save=args))


def parse_artifacts(
    node_artifacts: list[Any],
    artifacts: set[Artifact],
    target_node: PipelineNode,
    container_decls: dict[str, ContainerDeclaration],
):
    """
    Parse artifacts from the node artifacts list and populate the artifacts dictionary.
    """
    for artifact in node_artifacts:
        name = artifact["name"]
        container = artifact["container"]
        if container not in container_decls:
            raise ValueError(
                f"Artifact {name} references container {container} that is not "
                "declared in the pipeline"
            )
        artifacts.add(
            Artifact(
                name=name,
                node=target_node,
                container=container_decls[container],
                description=artifact.get("description"),
            )
        )


def parse_analyses(
    node_analyses: list[Any],
    analyses: set[AnalysisBinding],
    target_node: PipelineNode,
    container_decls: dict[str, ContainerDeclaration],
):
    """
    Parse analyses from the node analyses list and populate the analyses dictionary.
    """
    for analysis in node_analyses:
        analysis_name = analysis["analysis"]
        if analysis_name in analyses:
            raise ValueError(f"Analysis {analysis_name} is defined multiple times in the pipeline")
        containers = analysis["containers"]
        bindings = []
        for container in containers:
            if container not in container_decls:
                raise ValueError(
                    f"Analysis {analysis_name} references container {container} that is "
                    "not declared in the pipeline"
                )
            bindings.append(container_decls[container])

        analyses_registry = get_registry(Analysis)  # type: ignore[type-abstract]
        if analysis_name not in analyses_registry:
            raise ValueError(
                f"Analysis {analysis_name} is not registered, available analyses: "
                f"{sorted(analyses_registry.keys())}"
            )
        analysis_type: type[Analysis] = analyses_registry[analysis_name]

        name = analysis.get("name", analysis_name)

        analyses.add(
            AnalysisBinding(
                analysis=analysis_type(name),
                bindings=tuple(bindings),
                node=target_node,
            )
        )


def parse_task(
    task: Any,
    artifacts: set[Artifact],
    analyses: set[AnalysisBinding],
    container_decls: dict[str, ContainerDeclaration],
    parent: Optional[PipelineNode] = None,
) -> PipelineNode:
    """
    Parse a single task from the JSON value.
    The JSON value should contain a dictionary with the branch content.
    """

    # Create the PipelineNode
    res: PipelineNode
    if "savepoint" in task:
        res = parse_savepoint(
            task=task,
            container_decls=container_decls,
        )
    else:
        res = parse_pipe(
            task=task,
            container_decls=container_decls,
        )

    # Parse artifacts
    node_artifacts = task.get("artifacts", [])
    parse_artifacts(
        node_artifacts=node_artifacts,
        artifacts=artifacts,
        target_node=res,
        container_decls=container_decls,
    )

    # Parse analyses
    node_analyses = task.get("analyses", [])
    parse_analyses(
        node_analyses=node_analyses,
        analyses=analyses,
        target_node=res,
        container_decls=container_decls,
    )

    # Connect
    if parent is not None:
        parent.add_successor(res)
    return res


def parse_branch(
    node: Node,
    graph: dict[str, PipelineNode],
    artifacts: set[Artifact],
    analyses: set[AnalysisBinding],
    container_decls: dict[str, ContainerDeclaration],
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
        res = parse_task(
            task=task,
            artifacts=artifacts,
            analyses=analyses,
            container_decls=container_decls,
            parent=parent,
        )
        if head is None:
            head = res
        parent = res

    assert parent is not None, "A branch cannot be empty and not have a parent"
    assert head is not None, "A branch needs to have at least one task"
    return (head, parent)


def parse_branches(
    branches: dict[str, Any],
    artifacts: set[Artifact],
    analyses: set[AnalysisBinding],
    container_decls: dict[str, ContainerDeclaration],
) -> list[PipelineNode]:
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
        first_node, last_node = parse_branch(
            node=graph[name],
            graph=node_tips,
            container_decls=container_decls,
            artifacts=artifacts,
            analyses=analyses,
        )
        node_tips[name] = last_node
        if name in roots:
            result.append(first_node)

    return result


def parse_container_decls(
    containers: list[Any],
) -> dict[str, ContainerDeclaration]:
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


def schema() -> dict[str, Any]:
    """
    Return the jsonschema for the pipeline.
    """
    root = Path(__file__).resolve().parent
    with open(root / "schema.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline(values: Any) -> Pipeline:
    """
    Load a pipeline from parsed JSON / YAML / TOML.
    """
    # This setup is needed so we can have multiple schemas in the same file
    # so we can reuse the definitions
    full_schema = schema()
    resolver = jsonschema.RefResolver.from_schema(full_schema)
    pipeline_schema = full_schema["$defs"]["pipeline"]
    validator = jsonschema.Draft7Validator(pipeline_schema, resolver=resolver)
    validator.validate(values)

    # Parse create all the container declarations
    container_decls: dict[str, ContainerDeclaration] = parse_container_decls(values["containers"])

    # These will get filled while parsing branches
    artifacts: set[Artifact] = set()
    analyses: set[AnalysisBinding] = set()

    roots: list[PipelineNode] = parse_branches(
        branches=values["branches"],
        container_decls=container_decls,
        artifacts=artifacts,
        analyses=analyses,
    )

    if len(roots) == 1:
        root = roots[0]
    else:
        root = DummyPipelineNode("root")
        for node in roots:
            root.add_successor(node)

    return Pipeline(
        declarations=set(container_decls.values()),
        root=root,
        artifacts=artifacts,
        analyses=analyses,
    )


def load_pipeline_yaml(yaml_data: str) -> Pipeline:
    """
    Load a pipeline from a YAML string.
    """
    values = yaml.safe_load(yaml_data)
    return load_pipeline(values)


def load_pipeline_yaml_file(file: str) -> Pipeline:
    """
    Load a pipeline from a YAML file."""
    with open(file, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)
    return load_pipeline(values)
