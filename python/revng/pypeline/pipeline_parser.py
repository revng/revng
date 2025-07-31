#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jsonschema
import yaml

from revng.pypeline.container import Container, ContainerDeclaration
from revng.pypeline.pipeline import Analysis, AnalysisBinding, Artifact, Pipeline
from revng.pypeline.pipeline_node import PipelineNode
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.savepoint import SavePoint
from revng.pypeline.utils.registry import get_registry

JsonValue = Union[dict[str, "JsonValue"], list["JsonValue"], str, int, float, bool, None]
"""
This is called a json value, but represent some generic parsed data that
can be from json, yaml, toml, or any other similar format.
"""


@dataclass(slots=True)
class Node:
    """
    We use this to represent the graph of branches in the pipeline,
    so we can compute the loading order of the branches.
    """

    content: JsonValue
    is_root: bool = True
    successors: set[str] = field(default_factory=set)


def parse_pipe(
    task: dict[str, JsonValue],
    container_decls: dict[str, ContainerDeclaration],
):
    """Parse a pipe task from the JSON value."""
    pipes = get_registry(Pipe)
    pipe_name = task["pipe"]
    assert isinstance(pipe_name, str)
    if pipe_name not in pipes:
        raise ValueError(
            f"Pipe {pipe_name} is not registered, available pipes: " f"{sorted(pipes.keys())}"
        )
    pipe_ty = pipes[pipe_name]

    pipe_args = task.get("arguments", [])
    assert isinstance(pipe_args, list)
    bindings = []
    for arg in pipe_args:
        assert isinstance(arg, str)
        if arg not in container_decls:
            raise ValueError(
                f"While parsing {pipe_name}'s arguments found container `{arg}` "
                "that is not declared in the pipeline"
            )
        bindings.append(container_decls[arg])

    name = task.get("name")
    if name is not None:
        assert isinstance(name, str)
    configuration = task.get("configuration", "")
    assert isinstance(configuration, str)

    return PipelineNode(
        task=pipe_ty(
            name=name,
            static_configuration=configuration,
        ),
        bindings=bindings,
    )


def parse_savepoint(
    task: dict[str, JsonValue],
    container_decls: dict[str, ContainerDeclaration],
):
    """Parse a savepoint task from the JSON value."""
    assert isinstance(task["savepoint"], dict)
    name = task["savepoint"]["name"]
    assert isinstance(name, str)
    containers = task["savepoint"]["containers"]
    assert isinstance(containers, list)
    args = []
    for container_name in containers:
        assert isinstance(container_name, str), f"Container name {container_name} is not a string"
        if container_name not in container_decls:
            raise ValueError(f"Container {container_name} is not declared in the pipeline")
        args.append(container_decls[container_name])
    return PipelineNode(SavePoint(name=name, to_save=args))


def parse_artifacts(
    node_artifacts: list[JsonValue],
    artifacts: set[Artifact],
    target_node: PipelineNode,
    container_decls: dict[str, ContainerDeclaration],
):
    """
    Parse artifacts from the node artifacts list and populate the artifacts dictionary.
    """
    for artifact in node_artifacts:
        assert isinstance(artifact, dict)
        name = artifact["name"]
        assert isinstance(name, str)
        container = artifact["container"]
        assert isinstance(container, str)
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
            )
        )


def parse_analyses(
    node_analyses: list[JsonValue],
    analyses: set[AnalysisBinding],
    target_node: PipelineNode,
    container_decls: dict[str, ContainerDeclaration],
):
    """
    Parse analyses from the node analyses list and populate the analyses dictionary.
    """
    for analysis in node_analyses:
        assert isinstance(analysis, dict)
        analysis_name = analysis["analysis"]
        assert isinstance(analysis_name, str)
        if analysis_name in analyses:
            raise ValueError(f"Analysis {analysis_name} is defined multiple times in the pipeline")
        containers = analysis["containers"]
        assert isinstance(containers, list)
        bindings = []
        for container in containers:
            assert isinstance(container, str)
            if container not in container_decls:
                raise ValueError(
                    f"Analysis {analysis_name} references container {container} that is "
                    "not declared in the pipeline"
                )
            bindings.append(container_decls[container])

        analyses_registry = get_registry(Analysis)
        if analysis_name not in analyses_registry:
            raise ValueError(
                f"Analysis {analysis_name} is not registered, available analyses: "
                f"{sorted(analyses_registry.keys())}"
            )
        analysis_ty: type[Analysis] = analyses_registry[analysis_name]

        name = analysis.get("name", analysis_name)
        assert isinstance(name, str), f"Analysis name {name} is not a string"

        analyses.add(
            AnalysisBinding(
                analysis=analysis_ty(name),
                bindings=tuple(bindings),
                node=target_node,
            )
        )


def parse_task(
    task: JsonValue,
    artifacts: set[Artifact],
    analyses: set[AnalysisBinding],
    container_decls: dict[str, ContainerDeclaration],
    parent: Optional[PipelineNode] = None,
) -> PipelineNode:
    """
    Parse a single task from the JSON value.
    The JSON value should contain a dictionary with the branch content.
    """
    assert isinstance(task, dict)
    assert isinstance(task.get("name", ""), str)

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
    assert isinstance(node_artifacts, list)
    parse_artifacts(
        node_artifacts=node_artifacts,
        artifacts=artifacts,
        target_node=res,
        container_decls=container_decls,
    )

    # Parse analyses
    node_analyses = task.get("analyses", [])
    assert isinstance(node_analyses, list)
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
) -> PipelineNode:
    """
    Parse the branch and return the last node in the branch.
    """
    assert isinstance(node.content, dict)
    parent: Optional[PipelineNode] = None
    if "from" in node.content:
        assert isinstance(node.content["from"], str)
        parent_name = node.content["from"]
        if parent_name not in graph:
            raise ValueError(f"Branch {parent_name} is not defined in the pipeline")
        parent = graph[parent_name]

    # Parse the nodes
    tasks = node.content.get("tasks", [])
    assert isinstance(tasks, list)
    for task in tasks:
        assert isinstance(task, dict)
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
        parent = res

    assert parent is not None, "A branch cannot be empty and not have a parent"
    return parent


def parse_branches(
    branches: dict[str, JsonValue],
    artifacts: set[Artifact],
    analyses: set[AnalysisBinding],
    container_decls: dict[str, ContainerDeclaration],
) -> PipelineNode:
    """
    Parse the branches from the JSON value.
    The JSON value should contain a dictionary of branches with their names and tasks.
    """

    # Find the root branch
    assert isinstance(branches, dict)
    graph: dict[str, Node] = {}
    for name, branch in branches.items():
        assert isinstance(name, str)
        assert isinstance(branch, dict)
        node = Node(content=branch)
        graph[name] = node
        if "from" in branch:
            node.is_root = False
            assert isinstance(branch["from"], str)
            from_node = graph[branch["from"]]
            from_node.successors.add(name)

    # Check that there is only one root branch
    roots = sorted(name for name, node in graph.items() if node.is_root)
    if len(roots) != 1:
        raise ValueError(f"There should be exactly one root branch but found {roots}")
    root = roots[0]

    # Parse the branches in the correct order (DFS)
    result: dict[str, PipelineNode] = {}
    queue: list[str] = [root]
    root_node: PipelineNode | None = None
    while queue:
        name = queue.pop(0)
        node = graph[name]
        # parse the branch
        assert name not in result, f"Branch {name} is defined multiple times"
        result[name] = parse_branch(
            node=node,
            graph=result,
            container_decls=container_decls,
            artifacts=artifacts,
            analyses=analyses,
        )
        if root_node is None:
            root_node = result[name]
            # Result contains the END of each branch, so we need to find the root node
            # by following the predecessors
            while root_node.predecessors:
                root_node = root_node.predecessors[0]
        # Enqueue the successors
        for successor in node.successors:
            assert successor not in result, (
                f"The pipeline has to be a tree, not a DAG. Branch {successor} "
                "is defined multiple times"
            )
            queue.append(successor)
    # Return the root node
    assert root_node is not None, "The root node should be defined"
    return root_node


def parse_container_decls(
    containers: list[JsonValue],
) -> dict[str, ContainerDeclaration]:
    """
    Parse the container declarations from the JSON value.
    The JSON value should contain a list of containers with their names and types.
    """
    containers_registry = get_registry(Container)
    container_decls: dict[str, ContainerDeclaration] = {}
    for container in containers:
        assert isinstance(container, dict)
        assert isinstance(container["name"], str)
        assert isinstance(container["type"], str)

        name = container["name"]
        ty = container["type"]
        if ty not in containers_registry:
            raise ValueError(
                f"Container type {ty} is not registered, the available types "
                "are: {list(sorted(containers_registry.keys()))}"
            )

        container_decls[name] = ContainerDeclaration(
            name=name,
            container_type=containers_registry[ty],
        )
    return container_decls


def schema() -> dict[str, JsonValue]:
    """
    Return the jsonschema for the pipeline.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "pipeline_schema.yml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline(values: JsonValue) -> Pipeline:
    """
    Load a pipeline from parsed JSON / YAML / TOML.
    """
    validator = jsonschema.Draft7Validator(schema())
    validator.validate(values)
    # Yeah the validator already checks that everything is correct, but
    # mypy doesn't

    # Parse create all the container declarations
    assert isinstance(values, dict)
    assert isinstance(values["containers"], list)
    container_decls: dict[str, ContainerDeclaration] = parse_container_decls(values["containers"])

    # These will get filled while parsing branches
    artifacts: set[Artifact] = set()
    analyses: set[AnalysisBinding] = set()

    assert isinstance(values["branches"], dict)
    root: PipelineNode = parse_branches(
        branches=values["branches"],
        container_decls=container_decls,
        artifacts=artifacts,
        analyses=analyses,
    )
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
