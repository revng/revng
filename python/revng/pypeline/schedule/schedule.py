#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import Any, Dict, List, Optional, Set

import yaml

from revng.pypeline.container import ConfigurationId, ContainerDeclaration, ContainerSet
from revng.pypeline.graph import Graph
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.pipeline_node import PipelineConfiguration
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.task.pipe import Pipe
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.savepoint import SavePoint
from revng.pypeline.task.task import ObjectDependencies, TaskArgumentAccess
from revng.pypeline.utils.logger import pypeline_logger

from .scheduled_task import ScheduledTask


class Schedule:
    """
    A pipeline, given a set of requests, a model and the configuration of pipes, can produce a
    Schedule, i.e., a list (actually, a DAG) of tasks that need to be run with a certain set of
    requests in order to fulfill the requests.
    """

    def __init__(
        self,
        declarations: Set[ContainerDeclaration],
        root: ScheduledTask,
        pipeline_configuration: PipelineConfiguration,
    ):
        self.declarations = set(declarations)
        self.root = root
        self.tasks: Set[ScheduledTask] = set(root.all_dependencies())
        self.pipeline_configuration: PipelineConfiguration = pipeline_configuration

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
                for argument in node.node.arguments:
                    new_node.entries.append(argument.name)

                nodes_map[node] = new_node
                graph.nodes.add(new_node)

            return nodes_map[node]

        to_visit: List[ScheduledTask] = [self.root]
        visited: Set[ScheduledTask] = set()
        while to_visit:
            node = to_visit.pop()
            graph_node = get_node(node)
            node_inputs: List[ContainerDeclaration] = list(node.node.arguments)

            for predecessor in node.depends_on:
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

    def run(
        self,
        model: ReadOnlyModel,
        storage_provider: StorageProvider,
    ) -> ContainerSet:
        # Produce a set of working containers
        working_containers: ContainerSet = {
            declaration: declaration.instance() for declaration in self.declarations
        }

        ready: ScheduledTask | None = self._pick_task()

        while ready:
            pypeline_logger.log(f"Running {ready.node.task.name}")

            configuration: ConfigurationId = ready.node.configuration_id(
                self.pipeline_configuration
            )

            task_dependencies: ObjectDependencies | None = ready.run(
                model=model,
                containers=working_containers,
                pipeline_configuration=self.pipeline_configuration,
                storage_provider=storage_provider,
            )

            for declaration, container in sorted(
                working_containers.items(), key=lambda item: item[0].name
            ):
                pypeline_logger.log(f"  {declaration.name}: {str(container.objects())}")

            if isinstance(ready.node.task, Pipe):
                assert task_dependencies is not None
                assert (
                    ready.node.savepoint_range is not None
                ), "Savepoint range should be set for all Pipes"
                storage_provider.add_dependencies(
                    ready.node.savepoint_range, configuration, task_dependencies
                )
            else:
                assert task_dependencies is None

            ready = self._pick_task()

        return working_containers

    def _pick_task(self) -> Optional[ScheduledTask]:
        # TODO: use a graph
        for task in self.tasks:
            if task.completed:
                continue

            ready = True
            for dependency in task.depends_on:
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
            toposorter.add(task, *task.depends_on)

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
                        "depends_on": [visited_tasks.index(t) for t in task.depends_on],
                        "static_config": pipe.static_configuration,
                        "dynamic_config": self.pipeline_configuration.get(pipe, ""),
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
                            "configuration_hash": task.node.configuration_id(
                                self.pipeline_configuration
                            ),
                            "incoming": incoming,
                            "outgoing": outgoing,
                        }
                    )

                assert task.node.savepoint_range is not None
                tasks.append(
                    {
                        "type": "SavePoint",
                        "node_id": task.node.id,
                        "depends_on": [visited_tasks.index(t) for t in task.depends_on],
                        "name": savepoint.name,
                        "id": task.node.savepoint_range.start,
                        "containers": sp_containers,
                    }
                )
            else:
                raise ValueError(f"Unknown task: {type(task.node.task).__name__}")
            visited_tasks.append(task)

        return yaml.safe_dump({"containers": containers, "tasks": tasks})
