#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Generator, Iterable, List, Mapping, Optional, Set

import yaml

from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.utils import PypelineException

from .analysis import Analysis, AnalysisList
from .container import Container, ContainerDeclaration
from .graph import Graph
from .model import Model, ModelDiff, ReadOnlyModel
from .object import ObjectID, ObjectSet
from .pipeline_node import PipelineConfiguration, PipelineNode
from .schedule.schedule import Schedule
from .schedule.scheduled_task import ScheduledTask
from .storage.storage_provider import InvalidatedObjects, ObjectsToInvalidate, SavePointsRange
from .storage.storage_provider import StorageProvider
from .task.pipe import Pipe
from .task.requests import Requests
from .task.savepoint import SavePoint
from .task.task import TaskArgumentAccess
from .utils.default_dict_from_key import DefaultDictFromKey
from .utils.logger import pypeline_logger
from .utils.registry import get_singleton


@dataclass(frozen=True, slots=True)
class ArtifactCategory:
    name: str
    show_by_default: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "show_by_default": self.show_by_default}


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
    category: ArtifactCategory
    description: Optional[str] = None
    filename: str | None = None
    # TODO: this should be a property of the pipe that inserts the locations,
    # but for now we define it statically as a property of the artifact.
    defined_locations: list[str] = field(default_factory=list, hash=False)
    preferred_artifacts: list[str] = field(default_factory=list, hash=False)

    def is_cacheable(self) -> bool:
        """An artifact is cacheable if it's backed by a savepoint."""
        return isinstance(self.node.task, SavePoint)

    def pipe_dependencies(self) -> list[str]:
        return sorted({p.name for p in self.node.pipe_dependencies})

    def to_dict(self) -> dict:
        """Convert the artifact to a dictionary representation."""
        result = {
            "name": self.name,
            "container": self.container.name,
            "cacheable": self.is_cacheable(),
            "pipe_dependencies": self.pipe_dependencies(),
            "category": self.category.to_dict(),
            "defined_locations": self.defined_locations,
            "preferred_artifacts": self.preferred_artifacts,
        }
        if self.filename is not None:
            result["filename"] = self.filename
        return result


@dataclass(frozen=True, slots=True)
class AnalysisBinding:
    """Allows to bind an analysis to a pipeline node."""

    analysis: Analysis
    bindings: tuple[ContainerDeclaration, ...]
    node: PipelineNode

    def to_dict(self) -> dict:
        """Convert the data into a dictionary representation."""
        return {
            "name": self.analysis.name,
            "is_available": self.analysis.is_available(),
            "bindings": [
                {
                    "name": binding.name,
                    "container_type": binding.container_type.name,
                }
                for binding in self.bindings
            ],
            "node": self.node.id,
        }


class Pipeline:
    """
    A pipeline is a tree of tasks.

    Given a set of requests, a model and a configuration of the pipes, it can produce a schedule
    that fulfills the requests.
    """

    __slots__ = (
        "declarations",
        "root",
        "artifacts",
        "analyses",
        "analysis_lists",
        "savepoint_id_to_artifact",
        "savepoint_id_to_name",
    )

    def __init__(
        self,
        declarations: set[ContainerDeclaration],
        root: PipelineNode,
        artifacts: Optional[set[Artifact]] = None,
        analyses: Optional[set[AnalysisBinding]] = None,
        analysis_lists: Optional[Iterable[AnalysisList]] = None,
    ):
        self.root = root
        self.declarations = set(declarations)

        self.savepoint_id_to_artifact: dict[int, Artifact] = {}
        self.savepoint_id_to_name: dict[int, str] = {}

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

        self.analysis_lists: dict[str, AnalysisList] = {}
        """
        Aliases for lists of analyses to execute sequentially.
        """
        for analysis_list in analysis_lists or set():
            # Check that the given analysis list is valid
            if analysis_list.name in self.analyses:
                raise ValueError(
                    f"Analyses list alias {analysis_list.name} conflicts with an existing analysis"
                    " name"
                )
            for analysis_name in analysis_list.analyses:
                if analysis_name not in self.analyses:
                    raise ValueError(
                        f"Analysis {analysis_name} in analyses list {analysis_list.name} is not "
                        "defined in the pipeline"
                    )
            if analysis_list.name in self.analysis_lists:
                raise ValueError(
                    f"Analyses list alias {analysis_list.name} is defined multiple times in the "
                    "pipeline"
                )
            # Add the analysis list
            self.analysis_lists[analysis_list.name] = analysis_list

        Pipeline.assign_savepoint_ranges(root)

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
            if isinstance(node.task, SavePoint):
                assert node.savepoint_range is not None
                self.savepoint_id_to_name[node.savepoint_range.start] = node.task.name

        for name, artifact in self.artifacts.items():
            if name != artifact.name:
                raise ValueError(
                    f"Artifact name {artifact.name} does not match the key "
                    f"{name} in the artifacts map."
                )
            if isinstance(artifact.node.task, SavePoint):
                assert artifact.node.savepoint_range is not None
                self.savepoint_id_to_artifact[artifact.node.savepoint_range.start] = artifact

    @staticmethod
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
        # This is needed in the case the first node is a pipe, which is not a savepoint
        if isinstance(node.task, SavePoint):
            current_id += 1
        # Save the id on preorder visit
        start = current_id
        # Recurse on the children, but do it in a deterministic way
        end = current_id
        for child in node.sorted_successors():
            current_id = Pipeline.assign_savepoint_ranges(child, current_id)
            assert child.savepoint_range is not None, (
                f"Child {child.task.name} does not have a savepoint range assigned:"
                f" {child.savepoint_range}"
            )
            end = max(end, child.savepoint_range.end)
        # The end of the savepoint range is assigned on postorder visit.
        # Therefore the end is inclusive
        assert node.savepoint_range is None, (
            f"SavePoint {node.task.name} already has a savepoint range assigned"
            ": {node.savepoint_range}"
        )
        node.savepoint_range = SavePointsRange(start, end)
        # Return the new id
        return current_id

    def walk_pipeline(
        self, start: Optional[PipelineNode] = None, forward: bool = True, stable: bool = False
    ) -> Generator[PipelineNode, None, None]:
        """BFS walk of pipeline nodes"""
        assert int(not forward) + int(stable) < 2, "forward=False,stable=True is unsupported"

        to_visit: List[PipelineNode] = [start or self.root]
        visited: Set[PipelineNode] = set()

        if forward:

            def successors(node):
                return node.sorted_successors() if stable else node.successors

        else:

            def successors(node):
                return node.predecessors

        while len(to_visit) > 0:
            node = to_visit.pop()
            yield node
            visited.add(node)
            for child_node in successors(node):
                if child_node not in visited:
                    to_visit.append(child_node)

    def graph(self, container_edges=False) -> Graph:
        """A graph for debugging purposes."""

        graph = Graph()

        nodes_map: Dict[PipelineNode, Graph.Node] = {}

        access_to_str = {
            TaskArgumentAccess.READ: "R",
            TaskArgumentAccess.WRITE: "W",
            TaskArgumentAccess.READ_WRITE: "RW",
        }

        for node in self.walk_pipeline():
            if isinstance(node.task, Pipe):
                new_node = Graph.Node(node.task.name)
                for argument in node.arguments:
                    new_node.entries.append(f"{argument.name} [{access_to_str[argument.access]}]")
            else:
                new_node = Graph.Node(node.task.name, color="#2A52BE", bgcolor="#6C83BE")
                for argument in node.arguments:
                    new_node.entries.append(argument.name)

            nodes_map[node] = new_node
            graph.nodes.add(new_node)

            for predecessor in node.predecessors:
                graph.edges.add(
                    Graph.Edge(
                        nodes_map[predecessor],
                        new_node,
                        source_port=-1,
                        destination_port=-1,
                        style="bold",
                    )
                )

            if not container_edges:
                continue

            node_inputs: list[ContainerDeclaration] = list(node.argument_declarations)
            taken_inputs: set[int] = set()

            for parent_node in self.walk_pipeline(node, forward=False):
                if parent_node is node:
                    continue

                for source_index, parent_argument in enumerate(parent_node.arguments):
                    parent_container_decl = parent_argument.declaration()
                    if (
                        parent_argument.access == TaskArgumentAccess.READ
                        or parent_container_decl not in node_inputs
                    ):
                        continue

                    destination_index = node_inputs.index(parent_container_decl)
                    if destination_index in taken_inputs:
                        continue

                    taken_inputs.add(destination_index)
                    if parent_node in node.predecessors:
                        continue

                    graph.edges.add(
                        Graph.Edge(
                            nodes_map[parent_node],
                            new_node,
                            source_port=source_index,
                            destination_port=destination_index,
                            style="dashed",
                            color="#FD6D53",
                        )
                    )

        return graph

    def schedule(
        self,
        model: ReadOnlyModel,
        target_node: PipelineNode,
        requests: Requests,
        configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
    ) -> Schedule:
        tasks: DefaultDictFromKey[PipelineNode, ScheduledTask] = DefaultDictFromKey(
            lambda pn: ScheduledTask(pn, model, storage_provider, configuration)
        )
        # The pipeline is a tree, so we can just unroll the predecessors,
        # When we parallelize, we will make a subclass that overrides this method,
        # and probably it will first call it to produce the initial schedule and
        # then add the parallelization logic

        node: PipelineNode = target_node
        node_outgoing_requests = requests

        while not node_outgoing_requests.empty():
            pypeline_logger.debug_log(f"Scheduling node {node}")
            pypeline_logger.debug_log(f"Outgoing requests: {node_outgoing_requests}")
            orig = repr(node_outgoing_requests)
            # Each node should remove the requests it can handle
            # and add the requests it needs to satisfy the task
            node_ingoing_requests = node.prerequisites_for(
                model=model,
                requests=node_outgoing_requests,
                configuration=configuration,
                storage_provider=storage_provider,
            )
            assert orig == repr(node_outgoing_requests), (
                f"Node {node} modified the outgoing requests, which is not allowed. "
                f"Original: {orig}, modified: {node_outgoing_requests}"
            )
            pypeline_logger.debug_log(f"Computed Ingoing requests: {node_ingoing_requests}")
            # Store the computed requests so that we can use them in the run method
            tasks[node].add_requests(node_ingoing_requests, node_outgoing_requests)
            # If the node has no predecessors, we are done, but
            # we need to check that it has no requests left
            if not node.predecessors:
                assert node_ingoing_requests.empty(), (
                    f"Node {node} has no predecessors, but it still has "
                    f"requests: {node_ingoing_requests.minimize()}"
                )
                break

            assert len(node.predecessors) == 1, (
                f"Node {node} has multiple predecessors, but we assume a tree structure. "
                f"Predecessors: {node.predecessors}"
            )
            predecessor = node.predecessors[0]
            tasks[node].dependencies.append(tasks[predecessor])
            # Recurse on THE predecessor, its outgoing requests will be the
            # ingoing requests of the current node
            node = predecessor
            node_outgoing_requests = node_ingoing_requests

        # Here the task list has been decided, these are now iterated to apply
        # the following optimizations:
        # * Reduce the set of container declarations to those that are actually
        #   necessary to run the schedule
        # * Compact incoming and outgoing of each task to their reduced version
        # * Skip tasks where nothing in outgoing is actually written by the task
        #
        # These could be derived by using a dataflow-based approach to the
        # pipeline, where each pipe binds its inputs to the predecessor's pipe
        # output, but in the general case (with RW pipes) this turns into
        # having a mini-programming language and applying SSA analysis and the
        # like to deduplicate the containers (that, in the general case, need
        # to be duplicated at each node of the dataflow).
        scheduled_task: ScheduledTask | None = tasks[target_node]
        parent_scheduled_task: ScheduledTask | None = None
        used_declatations: set[str] = set()

        while scheduled_task is not None:
            # Assume that the schedule is a straight line, so a task has at
            # most one dependency
            assert len(scheduled_task.dependencies) in (0, 1)

            # Reduce incoming and outgoing
            scheduled_task.incoming = scheduled_task.incoming.minimize()
            scheduled_task.outgoing = scheduled_task.outgoing.minimize()

            # Figure out if a task will actually produce outputs
            written_out = Requests()
            for argument in scheduled_task.node.arguments:
                if argument.access != TaskArgumentAccess.READ:
                    container_decl = argument.declaration()
                    if container_decl in scheduled_task.outgoing:
                        written_out[container_decl] = scheduled_task.outgoing[container_decl]

            if written_out.empty() and parent_scheduled_task is not None:
                # If here the task does not actually need to produce anything
                # and can be skipped, by "glueing" its dependencies onto the parent
                assert [scheduled_task] == parent_scheduled_task.dependencies
                parent_scheduled_task.dependencies = scheduled_task.dependencies
            else:
                used_declatations.update(x.name for x in scheduled_task.node.argument_declarations)
                parent_scheduled_task = scheduled_task

            if len(scheduled_task.dependencies) == 1:
                scheduled_task = scheduled_task.dependencies[0]
            else:
                scheduled_task = None

        return Schedule(
            {v for v in self.declarations if v.name in used_declatations},
            tasks[target_node],
            configuration,
            model,
            storage_provider,
        )

    def get_artifact(
        self,
        model: ReadOnlyModel,
        artifact: Artifact,
        requests: ObjectSet,
        configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
        runner_context: RunnerContext = RunnerContext(),
    ) -> Container:
        schedule = self.schedule(
            model=model,
            target_node=artifact.node,
            requests=Requests({artifact.container: requests}),
            configuration=configuration,
            storage_provider=storage_provider,
        )
        return schedule.run(runner_context)[artifact.container]

    def run_analysis_list(
        self,
        model: ReadOnlyModel,
        analysis_list: AnalysisList,
        configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
        runner_context: RunnerContext = RunnerContext(),
    ) -> tuple[Model, InvalidatedObjects]:
        """
        Run a list of analyses on the pipeline, given a model and a set of requests.
        The analyses will return the new potentially modified model, and set it
        in the storage provider.
        """
        new_model: Model = model.clone()
        total_invalidated: InvalidatedObjects = InvalidatedObjects()

        pypeline_logger.debug_log(f"Running analysis list {analysis_list.name}")

        for analysis_name in analysis_list.analyses:
            if analysis_name not in self.analyses:
                raise PypelineException(f"Analysis {analysis_name} not found in the pipeline")

            if not self.analyses[analysis_name].analysis.is_available():
                raise PypelineException(
                    f"Analysis list {analysis_list.name} cannot be run "
                    f"because analysis {analysis_name} is not available"
                )

        for analysis_name in analysis_list.analyses:
            pypeline_logger.debug_log(f"Running analysis {analysis_name}")
            analysis = self.analyses[analysis_name]
            # Build the requests for the analysis
            requests = Requests()
            for container_decl in analysis.bindings:
                requests[container_decl] = model.all_objects(container_decl.container_type.kind)

            new_model, invalidated = self.run_analysis(
                model=model,
                analysis_name=analysis_name,
                requests=requests,
                configuration=configuration,
                storage_provider=storage_provider,
                runner_context=runner_context,
            )
            for location, objects in invalidated.items():
                if location in total_invalidated:
                    total_invalidated[location] = total_invalidated[location] | objects
                else:
                    total_invalidated[location] = objects
            model = ReadOnlyModel(new_model)

        return new_model, total_invalidated

    def run_analysis(
        self,
        model: ReadOnlyModel,
        analysis_name: str,
        requests: Requests,
        configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
        runner_context: RunnerContext = RunnerContext(),
    ) -> tuple[Model, InvalidatedObjects]:
        """
        Run an analysis on the pipeline, given a model and a set of requests.
        The analysis will return the new potentially modified model, and set it
        in the storage provider.
        """
        if analysis_name not in self.analyses:
            raise PypelineException(f"Analysis {analysis_name} not found in the pipeline")
        analysis_info = self.analyses[analysis_name]

        if len(requests) != len(analysis_info.bindings):
            raise PypelineException(
                f"Expected {len(analysis_info.bindings)} requests for analysis "
                f"{analysis_name}, but got {len(requests)}: {requests}"
            )

        for req in requests:
            if req not in analysis_info.bindings:
                raise PypelineException(
                    f"Request {req} but it's not compatible with in the "
                    f"analysis bindings: {analysis_info.bindings}"
                )

        if not analysis_info.analysis.is_available():
            raise PypelineException(f"Analysis {analysis_name} is not available")

        schedule = self.schedule(
            model=model,
            target_node=analysis_info.node,
            requests=requests,
            configuration=configuration,
            storage_provider=storage_provider,
        )
        all_containers = schedule.run(runner_context)

        new_model = runner_context.run_analysis(
            analysis=analysis_info.analysis,
            model=model,
            containers=[all_containers[decl] for decl in analysis_info.bindings],
            incoming=[requests.get(decl) for decl in analysis_info.bindings],
            configuration=configuration.get(analysis_info.analysis, ""),
        )

        diff = model.diff(ReadOnlyModel(new_model))
        custom_invalidated_objects = self._compute_custom_invalidation(
            configuration, storage_provider, diff
        )
        invalidated = storage_provider.invalidate(diff.paths(), custom_invalidated_objects)
        storage_provider.set_model(new_model)
        return new_model, invalidated

    def _compute_custom_invalidation(
        self,
        configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
        diff: ModelDiff,
    ) -> list[ObjectsToInvalidate]:
        result: list[ObjectsToInvalidate] = []
        for node in self.walk_pipeline():
            if not isinstance(node.task, Pipe):
                continue

            assert node.savepoint_range is not None

            # Optimization: since few pipes actually implement advanced
            # invalidation and it requires querying storage, check that the
            # method has actually been overridden.
            if not node.task.has_custom_invalidation():
                continue

            # Run the prelimiary check for the pipe
            if not node.task.requires_custom_invalidation(diff):
                continue

            # Fetch the custom invalidation data from storage
            configuration_id = node.configuration_id(configuration)
            invalidation_data = storage_provider.get_custom_invalidation_data(
                node.id, configuration_id
            )
            # If the data returned is empty, it means that:
            # * The pipe did not return any invalidation data
            # * The pipe returned it, but it happened to be empty
            # In both cases it means that the pipe is opting-out of custom invalidation
            if all(len(x) == 0 for x in invalidation_data):
                continue

            # Run the pipe's invalidate
            objects_to_invalidate = node.task.process_custom_invalidation(invalidation_data, diff)
            # Convert the invalidated objects in a pipeline-friendly format
            for index, objects in enumerate(objects_to_invalidate):
                container_decl = node.argument_declarations[index]
                result.append(
                    ObjectsToInvalidate(
                        node.savepoint_range, container_decl.name, configuration_id, objects
                    )
                )

        return result

    def deserialize_schedule(
        self, schedule: str, model: ReadOnlyModel, storage_provider: StorageProvider
    ) -> Schedule:
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

        obj_id_type = get_singleton(ObjectID)  # type: ignore[type-abstract]
        configuration: dict[Pipe | Analysis, str] = {}
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
                        container_kind, {obj_id_type.deserialize(x) for x in arg["incoming"]}
                    )
                    outgoing[container_declaration] = ObjectSet(
                        container_kind, {obj_id_type.deserialize(x) for x in arg["outgoing"]}
                    )
            elif task["type"] == "SavePoint":
                assert isinstance(pipeline_node.task, SavePoint)
                assert task["name"] == pipeline_node.task.name

                for container in task["containers"]:
                    container_declaration = container_map[container["name"]]
                    container_kind = container_declaration.container_type.kind

                    incoming[container_declaration] = ObjectSet(
                        container_kind, {obj_id_type.deserialize(x) for x in container["incoming"]}
                    )
                    outgoing[container_declaration] = ObjectSet(
                        container_kind, {obj_id_type.deserialize(x) for x in container["outgoing"]}
                    )
            else:
                raise ValueError(f"Unknown task type: \"{task['type']}\"")

            dependencies = [scheduled_tasks[i] for i in task["dependencies"]]
            scheduled_task = ScheduledTask(
                pipeline_node,
                model,
                storage_provider,
                configuration,
                (incoming, outgoing),
                dependencies,
            )
            scheduled_tasks.append(scheduled_task)

        return Schedule(declarations, scheduled_tasks[-1], configuration, model, storage_provider)
