#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from typing import Generator

from revng.pypeline.container import ContainerSet
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.pipeline_node import PipelineConfiguration, PipelineNode
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.task.pipe import ScheduledTaskDependencies
from revng.pypeline.task.requests import Requests


class ScheduledTask:
    """
    The scheduling works by figuring out which tasks to run, and which dependencies
    each task can fulfill.

    The schedule is stored as a Directed Acyclic Graph (DAG) of ScheduledTask objects.

    While the schedule is originally a path of the pipeline tree, we can apply
    transformations to it that can make it a DAG. An example of this is kind of
    transformations is the parallelization of tasks, where a ScheduleTask is
    split into multiple ScheduledTask objects that run in parallel the same task,
    with partitioned inputs and outputs requests.
    Moreover, using a DAG opens up the possibility of merging multiple schedules
    into a single one to improve performance of batch jobs.

    As Tasks should be side-effect free, multiple ScheduledTasks can share
    the same PipelineNode instance.
    """

    def __init__(
        self,
        node: PipelineNode,
        model: ReadOnlyModel,
        storage_provider: StorageProvider,
        pipeline_configuration: PipelineConfiguration,
        requests: tuple[Requests, Requests] | None = None,
        dependencies: list[ScheduledTask] | None = None,
    ):
        self.node = node
        """The node of the pipeline we scheduled."""
        self.model = model
        """The model that this task will run on."""
        self.storage_provider = storage_provider
        """The storage provider that the task will use."""
        self.pipeline_configuration = pipeline_configuration
        """The pipeline configuration"""

        self.completed: bool = False
        """This is used for debug, and asserting that a schedule is not run twice."""
        self.incoming = Requests()
        """
        These are the dependencies of the task, i.e. the objects that this task
        needs to run. This is computed during the scheduling phase, and is only used
        to check that the task can run.
        """
        if requests is not None:
            self.incoming = requests[0]

        self.outgoing = Requests()
        """These are the objects this task is supposed to compute and put in each container."""
        if requests is not None:
            self.outgoing = requests[1]

        self.dependencies: list[ScheduledTask] = []
        """
        These are the scheduled tasks that this task depends on, after the scheduling
        phase this list will contain only 0 or 1 elements as it's a path, but
        the schedule can become a DAG.
        """
        if dependencies is not None:
            self.dependencies = dependencies[:]

    def add_requests(self, incoming: Requests, outgoing: Requests) -> None:
        """
        Add requests to the incoming and outgoing requests of this task.
        """
        self.incoming.merge(incoming)
        self.outgoing.merge(outgoing)

    def run(self, containers: ContainerSet) -> ScheduledTaskDependencies | None:
        """
        Run the task with the requests computed during the scheduling phase.
        """
        if self.completed:
            raise RuntimeError(f"ScheduledTask {self.node} has already been run.")
        self.incoming.check(containers)
        result = self.node.run(
            model=self.model,
            containers=containers,
            incoming=self.incoming,
            outgoing=self.outgoing,
            pipeline_configuration=self.pipeline_configuration,
            storage_provider=self.storage_provider,
        )
        self.outgoing.check(containers)
        self.completed = True
        return result

    def all_dependencies(self) -> Generator[ScheduledTask, None, None]:
        """
        From this task, yield all the transitive dependencies of this task.
        This is used to get all the tasks involved in the execution of this task,
        including the task itself.
        """
        yield self
        for dependency in self.dependencies:
            yield from dependency.all_dependencies()

    def __hash__(self):
        """
        The intended use of this hash is to be used to do visits on the schedule graph.
        So this hash is just the id of the object, which is unique for each instance.
        Two instances of the same task will have different ids, and thus different hashes.
        """
        return hash(id(self))
