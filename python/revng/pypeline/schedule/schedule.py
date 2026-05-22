#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import Any, Dict, List, Optional, Set

import yaml

from revng.pypeline.container import ContainerDeclaration, ContainerSet
from revng.pypeline.graph import Graph
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.savepoint import SavePoint
from revng.pypeline.task.task import TaskArgumentAccess
from revng.pypeline.utils import PypelineException
from revng.pypeline.utils.logger import pypeline_logger

from .scheduled_task import PipeScheduledTask, SavepointScheduledTask, ScheduledTask


class Schedule:
    """
    A pipeline, given a set of requests, a model and the configuration of pipes, can produce a
    Schedule, i.e., a list (actually, a DAG) of tasks that need to be run with a certain set of
    requests in order to fulfill the requests.
    """

    def __init__(
        self,
        declarations: Set[ContainerDeclaration],
        target_task: ScheduledTask,
        model: ReadOnlyModel,
        storage_provider: StorageProvider,
    ):
        self.declarations = set(declarations)
        self.target_task = target_task
        self.tasks: Set[ScheduledTask] = set(target_task.all_dependencies())
        self.model = model
        self.storage_provider = storage_provider

    def graph(self) -> Graph:
        """
        Produce a graph for debugging purposes.
        """

        graph = Graph()

        nodes_map: Dict[ScheduledTask, Graph.Node] = {}

        def label(requests: Requests) -> str:
            result = ""
            for container, objects in requests.items():
                result += f"{container.name}:\n  " + "\n  ".join(str(x) for x in objects) + "\n"
            return result

        def get_node(node: ScheduledTask) -> Graph.Node:
            if node not in nodes_map:
                new_node = Graph.Node(node.node.task.name)
                if node.completed:
                    new_node.bgcolor = "lightgreen"
                for argument in node.node.argument_declarations:
                    new_node.entries.append(argument.name)

                nodes_map[node] = new_node
                graph.nodes.add(new_node)

            return nodes_map[node]

        to_visit: List[ScheduledTask] = [self.target_task]
        visited: Set[ScheduledTask] = set()
        while to_visit:
            node = to_visit.pop()
            graph_node = get_node(node)
            node_inputs: List[ContainerDeclaration] = list(node.node.argument_declarations)

            for predecessor in node.dependencies:
                for source_index, argument in enumerate(predecessor.node.task.arguments):
                    if argument.access == TaskArgumentAccess.READ or argument not in node_inputs:
                        continue

                    destination_index = node_inputs.index(argument)

                    source_node = get_node(predecessor)
                    new_edge = Graph.Edge(
                        source_node,
                        graph_node,
                        source_port=source_index,
                        destination_port=destination_index,
                        head_label=label(predecessor.outgoing),
                        tail_label=label(node.incoming),
                    )
                    graph.edges.add(new_edge)

                if predecessor not in visited:
                    to_visit.append(predecessor)
                    visited.add(predecessor)

        return graph

    def run(self, runner_context: RunnerContext = RunnerContext()) -> ContainerSet:
        for task in self.tasks:
            if isinstance(task.node.task, Pipe):
                try:
                    task.node.task.check_precondition(self.model)
                # TODO: eventually the pipe will raise `PypelineException` directly
                except RuntimeError as e:
                    raise PypelineException(
                        f"Preconditions were not met for pipe {task.node.task.name}: {e}"
                    )

        # Notify the tasks which containers are going to be discardable
        self._identify_discardable_containers()
        # Compute which savepoints are responsible for which pipes
        self._assign_responsible_pipes()

        # Produce a set of working containers
        working_containers: ContainerSet = {
            declaration: declaration.instance() for declaration in self.declarations
        }

        ready: ScheduledTask | None = self._pick_task()

        while ready is not None:
            pypeline_logger.debug_log(f"Running {ready.node.task.name}")

            ready.run(working_containers, runner_context)

            for declaration, container in sorted(
                working_containers.items(), key=lambda item: item[0].name
            ):
                pypeline_logger.debug_log(f"  {declaration.name}: {str(container.objects())}")

            ready = self._pick_task()

        return working_containers

    def _pick_task(self) -> Optional[ScheduledTask]:
        # TODO: use a graph
        for task in self.tasks:
            if task.completed:
                continue

            ready = True
            for dependency in task.dependencies:
                if not dependency.completed:
                    ready = False
                    break

            if ready:
                return task

        return None

    def serialize(self) -> str:
        """Serialize the Schedule to a YAML string"""

        containers = []
        for container in self.declarations:
            containers.append({"name": container.name, "type": container.container_type.__name__})

        toposorter: TopologicalSorter[ScheduledTask] = TopologicalSorter()
        for task in self.tasks:
            toposorter.add(task, *task.dependencies)

        tasks: list[Any] = []
        visited_tasks: list[ScheduledTask] = []
        for task in toposorter.static_order():
            if isinstance(task.node.task, Pipe):
                pipe: Pipe = task.node.task

                args = []
                for declaration in task.node.bindings:
                    incoming = [x.serialize() for x in task.incoming.get(declaration)]
                    outgoing = [x.serialize() for x in task.outgoing.get(declaration)]
                    args.append(
                        {"name": declaration.name, "incoming": incoming, "outgoing": outgoing}
                    )

                tasks.append(
                    {
                        "type": "Pipe",
                        "node_id": task.node.id,
                        "name": pipe.name,
                        "dependencies": [visited_tasks.index(t) for t in task.dependencies],
                        "static_config": pipe.static_configuration,
                        "dynamic_config": task.configuration.get(pipe, ""),
                        "args": args,
                    }
                )
            elif isinstance(task.node.task, SavePoint):
                savepoint = task.node.task

                sp_containers = []
                for declaration in self.declarations:
                    incoming = [x.serialize() for x in task.incoming.get(declaration)]
                    outgoing = [x.serialize() for x in task.outgoing.get(declaration)]
                    if len(incoming) == 0 and len(outgoing) == 0:
                        continue

                    sp_containers.append(
                        {
                            "name": declaration.name,
                            "configuration_hash": task.node.configuration_id(task.configuration),
                            "incoming": incoming,
                            "outgoing": outgoing,
                        }
                    )

                assert task.node.savepoint_range is not None
                tasks.append(
                    {
                        "type": "SavePoint",
                        "node_id": task.node.id,
                        "dependencies": [visited_tasks.index(t) for t in task.dependencies],
                        "name": savepoint.name,
                        "id": task.node.savepoint_range.start,
                        "containers": sp_containers,
                    }
                )
            else:
                raise ValueError(f"Unknown task: {type(task.node.task).__name__}")
            visited_tasks.append(task)

        return yaml.safe_dump({"containers": containers, "tasks": tasks})

    def _identify_discardable_containers(self):
        """
        Given a schedule, set the ScheduleTasks with the containers that will
        be discarded when the schedule is executed.
        """

        scheduled_task: ScheduledTask | None = self.target_task
        # Assume that all the outgoing request of the target task are going to
        # be read, so the caller of the `run` method is implicitly the last reader
        readers_encountered: set[ContainerDeclaration] = {
            cd for cd, objects in self.target_task.outgoing.items() if len(objects) != 0
        }

        # Inspect the tasks in backward order, from the last one to the first.
        # The logic used is the following:
        # * If a task is the last one to read (either with READ or READ_WRITE)
        #   a container, the container should be marked as disposable
        # * If a task clobbers a container (with WRITE) then the container can
        #   be marked as disposable in the preceding task
        while scheduled_task is not None:
            # Assume that this schedule is a straight line of tasks, as it
            # simplifies the logic needed
            assert len(scheduled_task.dependencies) in (0, 1)

            for argument in scheduled_task.node.arguments:
                container_declaration = argument.declaration()
                # Here check for both READ and READ_WRITE, since if it's the
                # last one the READ_WRITE is effectively READ
                if (
                    argument.access & TaskArgumentAccess.READ
                    and container_declaration not in readers_encountered
                ):
                    scheduled_task.disposable_containers.add(container_declaration)
                    readers_encountered.add(container_declaration)

                # If the task writes to the container (clobbering it) then
                # tasks that depend on this one can also expire the container
                if argument.access == TaskArgumentAccess.WRITE:
                    readers_encountered.discard(container_declaration)

            if len(scheduled_task.dependencies) == 1:
                scheduled_task = scheduled_task.dependencies[0]
            else:
                scheduled_task = None

    def _assign_responsible_pipes(self):
        """
        Given a schedule, assign to the SavepointScheduledTask the pipes that
        it's responsible for. This is needed because savepoints are responsible
        for uploading both the data in the container and the pipes that are
        responsible for. This minimizes transfers and avoids dependencies
        referencing objects that are not present in storage.
        """

        sorter: TopologicalSorter[ScheduledTask] = TopologicalSorter()
        for task in self.tasks:
            sorter.add(task, *task.dependencies)

        pipes_dependencies: dict[ScheduledTask, list[PipeScheduledTask]] = {}
        for task in sorter.static_order():
            pipe_dependencies = []
            for task_dependency in task.dependencies:
                if isinstance(task_dependency, PipeScheduledTask):
                    pipe_dependencies.append(task_dependency)
                    pipe_dependencies.extend(pipes_dependencies[task_dependency])

            if isinstance(task, PipeScheduledTask):
                pipes_dependencies[task] = pipe_dependencies
            elif isinstance(task, SavepointScheduledTask):
                task.dependant_pipes = pipe_dependencies
            else:
                raise ValueError
