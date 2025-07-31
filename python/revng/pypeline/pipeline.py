#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Generator, Generic, List, Mapping, Optional, Set, TypeVar

import yaml

from revng.pypeline.container import Container, ContainerDeclaration
from revng.pypeline.graph import Graph
from revng.pypeline.model import Model, ReadOnlyModel
from revng.pypeline.object import ObjectID, ObjectSet
from revng.pypeline.pipeline_node import PipelineConfiguration, PipelineNode
from revng.pypeline.schedule.schedule import Schedule
from revng.pypeline.schedule.scheduled_task import ScheduledTask
from revng.pypeline.storage.storage_provider import SavePointsRange, StorageProvider
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.savepoint import SavePoint
from revng.pypeline.task.task import TaskArgumentAccess
from revng.pypeline.utils.default_dict_from_key import DefaultDictFromKey
from revng.pypeline.utils.registry import registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Artifact:
    """
    An artifact is a container in a certain point of the pipeline with some extra
    metadata, such as a name.

    It's designed to mark interesting results in the pipeline, so that users an
    easily obtain them.
    """

    name: str
    node: PipelineNode
    container: ContainerDeclaration


@registry
class Analysis(ABC):
    """
    An analysis makes changes to the model. In order to do this, it might inspect
    previously produced results of the pipeline.

    An analysis is the way in which users are expected to make changes to the model.
    The changes applied by a run of an analysis might lead to invalidate certain
    objects in save points.
    """

    __slots__: tuple = ("name",)

    def __init__(self, name: str):
        """
        Initialize the analysis with a name.
        The name is used for debugging and logging purposes.
        """
        self.name = name

    @classmethod
    @abstractmethod
    def signature(cls) -> tuple[type[Container], ...]:
        """
        The containers required by the analysis, it needs to be a class property
        to auto-generate the CLI commands.
        """

    @abstractmethod
    def run(
        self,
        model: Model,
        containers: list[Container],
        incoming: list[ObjectSet],
        configuration: str,
    ):
        """
        Run the analysis on the model, using the containers and
        incoming requests. The analysis will modify inplace the
        model so you must make a copy before running it, so you
        can compute the diff for invalidation purposes.
        """


@dataclass(frozen=True, slots=True)
class AnalysisBinding:
    """Allows to bind an analysis to a pipeline node."""

    analysis: Analysis
    bindings: tuple[ContainerDeclaration, ...]
    node: PipelineNode


C = TypeVar("C", bound=Model)


def assign_savepoint_ranges(node: PipelineNode, current_id: int = 0) -> int:
    """
    Assigns savepoint ranges to the nodes in the pipeline tree.
    A savepoint range is a pair of integers that represent the start and end of
    the savepoint in a subtree. The idea is to deduplicate things common to a
    savepoint and all its children, so that we can efficiently represent a
    subtree of savepoints as a continuous range of integers.
    """
    # We ID only the savepoints, so we increment the ID only if the current node is a savepoint.
    # This implies that, while the ids start at 0, the first savepoint will have id 1.
    # This is needed in the case the first node is a pipe, which is not a savepoint.
    if isinstance(node.task, SavePoint):
        current_id += 1
    # Save the id on preorder visit
    start = current_id
    # Recurse on the children, but do it in a deterministic way.
    end = current_id
    for child in node.sorted_successors():
        current_id = assign_savepoint_ranges(child, current_id)
        assert child.savepoint_range is not None, (
            f"Child {child.task.name} does not have a savepoint range assigned:"
            f" {child.savepoint_range}"
        )
        end = max(end, child.savepoint_range.end)
    # The end of the savepoint range is assigned on postorder visit.
    # Therefore the end is inclusive.
    assert node.savepoint_range is None, (
        f"SavePoint {node.task.name} already has a savepoint range assigned"
        ": {node.savepoint_range}"
    )
    node.savepoint_range = SavePointsRange(start, end)
    # Return the new id
    return current_id


class Pipeline(Generic[C]):
    """
    A pipeline is a tree of tasks.

    Given a set of requests, a model and a configuration of the pipes, it can produce a schedule
    that fulfills the requests.
    """

    __slots__ = ("declarations", "root", "artifacts", "analyses")

    def __init__(
        self,
        declarations: Set[ContainerDeclaration],
        root: PipelineNode,
        artifacts: Optional[set[Artifact]] = None,
        analyses: Optional[set[AnalysisBinding]] = None,
    ):
        self.root = root
        self.declarations = set(declarations)

        self.artifacts: Mapping[str, Artifact] = {}
        """
        The artifacts, indexed by their name for easy access.
        """
        for artifact in artifacts or set():
            if artifact.name in self.artifacts:
                raise ValueError(
                    f"Artifact {artifact.name} is defined multiple times in the pipeline"
                )
            self.artifacts[artifact.name] = artifact

        self.analyses: Mapping[str, AnalysisBinding] = {}
        """
        The analyses, indexed by their name for easy access.
        """
        for analysis in analyses or set():
            if analysis.analysis.name in self.analyses:
                raise ValueError(
                    f"Analysis {analysis.analysis.name} is defined multiple times in the pipeline"
                )
            self.analyses[analysis.analysis.name] = analysis

        assign_savepoint_ranges(root)

        id_index = 0
        # Set the dependencies field
        for node in self.walk_pipeline(stable=True):
            node.id = id_index
            id_index += 1

            if node is self.root:
                node.pipe_dependencies = set()
                continue

            node.pipe_dependencies = set(
                chain.from_iterable(n.pipe_dependencies for n in node.predecessors)
            )
            if isinstance(node.task, Pipe):
                node.pipe_dependencies.add(node.task)

    def __post_init__(self):
        for name, artifact in self.artifacts.items():
            if name != artifact.name:
                raise ValueError(
                    f"Artifact name {artifact.name} does not match the key "
                    f"{name} in the artifacts map."
                )

    def walk_pipeline(
        self, start: Optional[PipelineNode] = None, forward: bool = True, stable: bool = False
    ) -> Generator[PipelineNode, None, None]:
        """BFS walk of pipeline nodes"""
        to_visit: List[PipelineNode] = [start or self.root]
        visited: Set[PipelineNode] = set()

        if forward:

            def successors(node):
                if not stable:
                    return node.successors
                else:
                    return node.sorted_successors()

        else:

            def successors(node):
                assert not stable
                return node.predecessors

        while len(to_visit) > 0:
            node = to_visit.pop()
            yield node
            visited.add(node)
            for child_node in successors(node):
                if child_node not in visited:
                    to_visit.append(child_node)

    def graph(self) -> Graph:
        """A graph for debugging purposes."""

        graph = Graph()

        nodes_map: Dict[PipelineNode, Graph.Node] = {}

        def get_node(node: PipelineNode) -> Graph.Node:
            if node not in nodes_map:
                new_node = Graph.Node(node.task.name)
                for argument in node.arguments:
                    new_node.entries.append(argument.name)

                nodes_map[node] = new_node
                graph.nodes.add(new_node)

            return nodes_map[node]

        for node in self.walk_pipeline():
            graph_node = get_node(node)
            node_inputs: List[ContainerDeclaration] = list(node.arguments)

            for predecessor in node.predecessors:
                for source_index, argument in enumerate(predecessor.task.arguments):
                    if argument.access == TaskArgumentAccess.READ or argument not in node_inputs:
                        continue

                    destination_index = node_inputs.index(argument)
                    graph.edges.add(
                        Graph.Edge(
                            get_node(predecessor),
                            graph_node,
                            source_port=source_index,
                            destination_port=destination_index,
                        )
                    )

        return graph

    def schedule(
        self,
        model: ReadOnlyModel,
        target_node: PipelineNode,
        requests: Requests,
        pipeline_configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
    ) -> Schedule:
        tasks: DefaultDictFromKey[PipelineNode, ScheduledTask] = DefaultDictFromKey(ScheduledTask)
        # The pipeline is a tree, so we can just unroll the predecessors,
        # When we parallelize, we will make a subclass that overrides this method,
        # and probably it will first call it to produce the initial schedule and
        # then add the parallelization logic.

        node: PipelineNode = target_node
        node_outgoing_requests = requests

        while not node_outgoing_requests.empty():
            logger.debug("Scheduling node %s", node)
            logger.debug("Outgoing requests: %s", node_outgoing_requests)
            orig = repr(node_outgoing_requests)
            # each node should remove the requests it can handle
            # and add the requests it needs to satisfy the task
            node_ingoing_requests = node.prerequisites_for(
                model=model,
                requests=node_outgoing_requests,
                pipeline_configuration=pipeline_configuration,
                storage_provider=storage_provider,
            )
            assert orig == repr(node_outgoing_requests), (
                f"Node {node} modified the outgoing requests, which is not allowed. "
                f"Original: {orig}, modified: {node_outgoing_requests}"
            )
            logger.debug("Computed Ingoing requests: %s", node_ingoing_requests)
            # Store the computed requests so that we can use them in the run method
            tasks[node].request(node_ingoing_requests, node_outgoing_requests)
            # If the node has no predecessors, we are done, but
            # we need to check that it has no requests left.
            if not node.predecessors:
                assert node_ingoing_requests.empty(), (
                    f"Node {node} has no predecessors, but it still has "
                    f"requests: {node_ingoing_requests}"
                )
                break

            assert len(node.predecessors) == 1, (
                f"Node {node} has multiple predecessors, but we assume a tree structure. "
                f"Predecessors: {node.predecessors}"
            )
            predecessor = node.predecessors[0]
            tasks[node].depends_on.append(tasks[predecessor])
            # Recurse on THE predecessor, its outgoing requests will be the
            # ingoing requests of the current node.
            node = predecessor
            node_outgoing_requests = node_ingoing_requests

        return Schedule(self.declarations, tasks[target_node], pipeline_configuration)

    def get_artifact(
        self,
        model: ReadOnlyModel,
        artifact: Artifact,
        requests: ObjectSet,
        pipeline_configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
    ) -> Container:
        schedule = self.schedule(
            model=model,
            target_node=artifact.node,
            requests=Requests({artifact.container: requests}),
            pipeline_configuration=pipeline_configuration,
            storage_provider=storage_provider,
        )
        return schedule.run(
            model=model,
            storage_provider=storage_provider,
        )[artifact.container]

    def run_analysis(
        self,
        model: ReadOnlyModel,
        analysis_name: str,
        requests: Requests,
        analysis_configuration: str,
        pipeline_configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
    ) -> Model:
        if analysis_name not in self.analyses:
            raise ValueError(f"Analysis {analysis_name} not found in the pipeline")
        analysis_info = self.analyses[analysis_name]

        if len(requests) != len(analysis_info.bindings):
            raise ValueError(
                f"Expected {len(analysis_info.bindings)} requests for analysis "
                f"{analysis_name}, but got {len(requests)}: {requests}"
            )

        for req in requests:
            if req not in analysis_info.bindings:
                raise ValueError(
                    f"Request {req} but it's not compatible with in the "
                    f"analysis bindings: {analysis_info.bindings}"
                )

        schedule = self.schedule(
            model=model,
            target_node=analysis_info.node,
            requests=requests,
            pipeline_configuration=pipeline_configuration,
            storage_provider=storage_provider,
        )
        all_containers = schedule.run(model=model, storage_provider=storage_provider)
        # The analysis modifies the model, but we need to invalidate
        # the changes, so we make a clone of the model.
        # This also allows to keep the original model intact
        # in case the analysis fails.
        new_model = model.clone()
        analysis_info.analysis.run(
            model=new_model,
            containers=[all_containers[decl] for decl in analysis_info.bindings],
            incoming=[
                requests.get(decl, ObjectSet(decl.container_type.kind, set()))
                for decl in analysis_info.bindings
            ],
            configuration=analysis_configuration,
        )
        storage_provider.invalidate(ReadOnlyModel(new_model).diff(model))
        # TODO: call storage_provider.set_model?
        return new_model

    def deserialize_schedule(self, schedule: str) -> Schedule:
        schedule_dict = yaml.safe_load(schedule)
        declarations = set()
        for container in schedule_dict["containers"]:
            for declaration in self.declarations:
                if container["name"] == declaration.name:
                    declarations.add(declaration)
                    break
            else:
                raise ValueError()

        pipeline_nodes = list(self.walk_pipeline(stable=True))
        container_map = {x.name: x for x in self.declarations}

        configuration = {}
        scheduled_tasks: list[ScheduledTask] = []
        for task in schedule_dict["tasks"]:
            pipeline_node: PipelineNode = pipeline_nodes[task["node_id"]]
            outgoing = Requests()
            incoming = Requests()

            if task["type"] == "Pipe":
                assert isinstance(pipeline_node.task, Pipe)
                assert task["name"] == pipeline_node.task.name

                configuration[pipeline_node.task] = task["dynamic_config"]
                for index, arg in enumerate(task["args"]):
                    container_declaration = pipeline_node.bindings[index]
                    assert container_declaration.name == arg["name"]
                    container_kind = container_declaration.container_type.kind

                    incoming[container_declaration] = ObjectSet(
                        container_kind, {ObjectID.deserialize_any(x) for x in arg["incoming"]}
                    )
                    outgoing[container_declaration] = ObjectSet(
                        container_kind, {ObjectID.deserialize_any(x) for x in arg["outgoing"]}
                    )
            elif task["type"] == "SavePoint":
                assert isinstance(pipeline_node.task, SavePoint)
                assert task["name"] == pipeline_node.task.name

                for container in task["containers"]:
                    container_declaration = container_map[container["name"]]
                    container_kind = container_declaration.container_type.kind

                    incoming[container_declaration] = ObjectSet(
                        container_kind, {ObjectID.deserialize_any(x) for x in container["incoming"]}
                    )
                    outgoing[container_declaration] = ObjectSet(
                        container_kind, {ObjectID.deserialize_any(x) for x in container["outgoing"]}
                    )
            else:
                raise ValueError(f"Unknown task type: {task['type']}")

            depends_on = [scheduled_tasks[i] for i in task["depends_on"]]
            scheduled_task = ScheduledTask(pipeline_node, False, outgoing, incoming, depends_on)
            scheduled_tasks.append(scheduled_task)

        return Schedule(declarations, scheduled_tasks[-1], configuration)
