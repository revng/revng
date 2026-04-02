#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, cast

from revng.pypeline.container import ContainerDeclaration, ContainerSet
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.pipeline_node import PipelineConfiguration, PipelineNode
from revng.pypeline.runner_context import RunnerContext
from revng.pypeline.storage.storage_provider import StorageProvider, StorageProviderFileProvider
from revng.pypeline.task.pipe import ObjectDependencies, Pipe, ScheduledTaskDependencies
from revng.pypeline.task.requests import Requests
from revng.pypeline.task.savepoint import SavePoint
from revng.pypeline.task.task import TaskArgumentAccess


class ScheduledTask(ABC):
    """
    The scheduling works by figuring out which tasks to run, and which dependencies
    each task can fulfill.

    The schedule is stored as a Directed Acyclic Graph (DAG) of
    ScheduledTask objects (which are specialized for the task that will be
    executed: PipeScheduledTask and SavepointScheduledTask).

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
        configuration: PipelineConfiguration,
        requests: tuple[Requests, Requests] | None = None,
        dependencies: list[ScheduledTask] | None = None,
    ):
        self.node = node
        """The node of the pipeline we scheduled."""
        self.model = model
        """The model that this task will run on."""
        self.storage_provider = storage_provider
        """The storage provider that the task will use."""
        self.configuration = configuration
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

        self.disposable_containers: set[ContainerDeclaration] = set()
        """
        These containers are expring, meaning that they will not be read after
        this task has run
        """

    def add_requests(self, incoming: Requests, outgoing: Requests) -> None:
        """
        Add requests to the incoming and outgoing requests of this task.
        """
        self.incoming.merge(incoming)
        self.outgoing.merge(outgoing)

    @abstractmethod
    def _run_impl(
        self, containers: ContainerSet, runner_context: RunnerContext
    ) -> ScheduledTaskDependencies | None: ...

    def run(
        self, containers: ContainerSet, runner_context: RunnerContext
    ) -> ScheduledTaskDependencies | None:
        """
        Run the task with the requests computed during the scheduling phase.
        """
        if self.completed:
            raise RuntimeError(f"ScheduledTask {self.node} has already been run.")

        self.incoming.check(containers)

        for container_declaration in self.disposable_containers:
            containers[container_declaration].set_is_disposable()

        result = self._run_impl(containers, runner_context)

        self.outgoing.check(containers)
        self.completed = True
        for container_declaration in self.disposable_containers:
            containers[container_declaration].dispose_if_possible()

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


class PipeScheduledTask(ScheduledTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.node.task, Pipe)

    def _run_impl(
        self, containers: ContainerSet, runner_context: RunnerContext
    ) -> ScheduledTaskDependencies:
        task: Pipe = cast(Pipe, self.node.task)
        bindings = self.node.bindings
        pipe_containers = [containers[decl] for decl in bindings]
        pipe_incoming = [self.incoming.get(decl) for decl in bindings]
        pipe_outgoing = [self.outgoing.get(decl) for decl in bindings]
        configuration = self.configuration.get(task, "")

        pipe_output = runner_context.run_pipe(
            pipe=task,
            file_provider=StorageProviderFileProvider(self.storage_provider),
            model=self.model,
            containers=pipe_containers,
            incoming=pipe_incoming,
            outgoing=pipe_outgoing,
            configuration=configuration,
        )

        for index, decl in enumerate(bindings):
            containers[decl] = pipe_containers[index]

        result_pipe: ObjectDependencies = []
        for index, index_deps in enumerate(pipe_output.dependencies):
            container_type = task.signature()[index]
            if container_type.access == TaskArgumentAccess.READ:
                assert len(index_deps) == 0, (
                    "An read only container cannot produce new objects so it can't add "
                    f"dependencies. For container {container_type.name} got dependencies "
                    f"{index_deps}"
                )
            result_pipe.extend(
                (self.node.bindings[index].name, obj, path) for obj, path in index_deps
            )

        # Check if the pipe output
        if not all(len(x) == 0 for x in pipe_output.custom_invalidation):
            assert task.has_custom_invalidation(), (
                f"Pipe {task.name} returned advanced invalidation data"
                "but did not override the 'invalidate' method"
            )
            for index, invalidation in enumerate(pipe_output.custom_invalidation):
                argument = task.signature()[index]
                if argument.access == TaskArgumentAccess.READ:
                    assert len(invalidation) == 0, (
                        f"Pipe {task.name} returned advanced "
                        "invalidation data for a read-only container"
                    )

        return ScheduledTaskDependencies(result_pipe, pipe_output.custom_invalidation)


class SavepointScheduledTask(ScheduledTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.node.task, SavePoint)

    def _run_impl(self, containers: ContainerSet, runner_context: RunnerContext) -> None:
        task: SavePoint = cast(SavePoint, self.node.task)
        assert (
            self.node.savepoint_range is not None
        ), "SavePoint range must be set before calling run on a SavePoint"
        task.run(
            containers=containers,
            incoming=self.incoming,
            outgoing=self.outgoing,
            configuration_id=self.node.configuration_id(self.configuration),
            storage_provider=self.storage_provider,
            savepoint_range=self.node.savepoint_range,
        )
