#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

from revng.pypeline.container import ContainerSet
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.pipeline_node import PipelineConfiguration, PipelineNode
from revng.pypeline.storage.storage_provider import StorageProvider
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.task import ObjectDependencies


@dataclass(slots=True)
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

    node: PipelineNode
    """The node of the pipeline we scheduled."""

    completed: bool = False
    """This is used for debug, and asserting that a schedule is not run twice."""

    outgoing: Requests = field(default_factory=Requests)
    """These are the objects this task is supposed to compute and put in each container."""

    incoming: Requests = field(default_factory=Requests)
    """
    These are the dependencies of the task, i.e. the objects that this task
    needs to run. This is computed during the scheduling phase, and is only used
    to check that the task can run.
    """

    depends_on: list[ScheduledTask] = field(default_factory=list)
    """
    These are the scheduled tasks that this task depends on, after the scheduling
    phase this list will contain only 0 or 1 elements as it's a path, but
    the schedule can become a DAG.
    """

    def request(self, incoming: Requests, outgoing: Requests) -> None:
        """
        Add requests to the incoming and outgoing requests of this task.
        """
        self.incoming.merge(incoming)
        self.outgoing.merge(outgoing)

    def run(
        self,
        model: ReadOnlyModel,
        containers: ContainerSet,
        pipeline_configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
    ) -> ObjectDependencies | None:
        """
        Run the task with the requests computed during the scheduling phase.
        """
        if self.completed:
            raise RuntimeError(f"ScheduledTask {self.node} has already been run.")
        self.incoming.check(containers)
        result = self.node.run(
            model=model,
            containers=containers,
            incoming=self.incoming,
            outgoing=self.outgoing,
            pipeline_configuration=pipeline_configuration,
            storage_provider=storage_provider,
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
        for dependency in self.depends_on:
            yield from dependency.all_dependencies()

    def __hash__(self):
        """
        The intended use of this hash is to be used to do visits on the schedule graph.
        So this hash is just the id of the object, which is unique for each instance.
        Two instances of the same task will have different ids, and thus different hashes.
        """
        return hash(id(self))
