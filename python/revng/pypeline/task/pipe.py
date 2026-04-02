#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from collections.abc import Buffer, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Optional, final

from revng.pypeline.container import Configuration, Container, ContainerID
from revng.pypeline.model import ModelDiff, ModelPath, ReadOnlyModel
from revng.pypeline.object import ObjectID, ObjectSet
from revng.pypeline.storage.file_provider import FileProvider, FileRequest
from revng.pypeline.utils.cabc import ABC, abstractmethod

from .task import TaskArgument, TaskArgumentAccess

PipeObjectDependencies = Annotated[
    list[list[tuple[ObjectID, ModelPath]]],
    """
    A list representing the dependencies between the an object (in a certain
    container) produced by a Pipe. As the Pipe doesn't know the container
    names, it just returns the index of the container in the Pipe's signature.
    And then it's up to `PipelineNode` to remap the index to the container name.
    """,
]

PipeCustomInvalidation = Annotated[
    Sequence[Sequence[tuple[ObjectID, Buffer]]],
    """
    Additional opaque invalidation data returned by the Pipe, this will be fed
    back to the pipe's `invalidation` method to gather a list of `ObjectID`s to
    additionally purge.
    """,
]


@dataclass
class PipeDependencies:
    dependencies: PipeObjectDependencies
    custom_invalidation: PipeCustomInvalidation = field(default_factory=list)


ObjectDependencies = Annotated[
    list[tuple[ContainerID, ObjectID, ModelPath]],
    """
    A list representing the dependencies between the an object (in a certain container) produced
    by a certain task and the model.
    """,
]


@dataclass
class ScheduledTaskDependencies:
    dependencies: ObjectDependencies
    custom_invalidation: PipeCustomInvalidation = field(default_factory=list)


class Pipe(ABC):
    """
    A Pipe is a task that, given some input objects, a configuration string and the model, produces
    some new objects.
    """

    name: str

    @classmethod
    @abstractmethod
    def signature(cls) -> tuple[TaskArgument, ...]:
        """
        While tasks can have a dynamic arguments, like a savepoint
        can save different type of containers, a pipe has to have a static ones
        that do not depend on the instance. This could be a class attribute,
        but `@abstractclassattributes` does not exist in python.
        """

    @classmethod
    def static_configuration_help(cls) -> Optional[str]:
        """
        The text to display when the user asks for help on the
        static configuration of this task.
        Do not implement it, or return None, if the static configuration
        argument should not be added.
        """

    def __init__(self, static_configuration: str = ""):
        self.static_configuration: str = static_configuration

    @property
    def arguments(self) -> list[TaskArgument]:
        """
        Return the arguments of this pipe, which are the static configuration and the inputs and
        outputs.
        """
        return list(self.signature())

    @property
    def inputs(self) -> list[TaskArgument]:
        """
        Return the inputs of this pipe, which are the arguments that are not outputs.
        """
        return [arg for arg in self.signature() if arg.access != TaskArgumentAccess.WRITE]

    @property
    def outputs(self) -> list[TaskArgument]:
        """
        Return the outputs of this pipe, which are the arguments that are not inputs.
        """
        return [arg for arg in self.signature() if arg.access != TaskArgumentAccess.READ]

    @final
    def prerequisites_for(
        self,
        model: ReadOnlyModel,
        requests: list[ObjectSet],
    ) -> list[ObjectSet]:
        """
        Given a set of requests, a configuration and a model, produce a new set
        of requests that are required in order to run this pipeline successfully.
        """
        # List of empty requests, one per argument
        result = [ObjectSet(decl.container_type.kind, set()) for decl in self.arguments]

        # Cross-contaminate inputs and outputs
        for idx, decl in enumerate(self.arguments):
            # We must fill the readable containers
            if decl.access == TaskArgumentAccess.WRITE:
                continue

            for object_list in requests:
                result[idx].update(
                    model.move_to_kind(
                        object_list,
                        decl.container_type.kind,
                    )
                )

        return result

    def check_precondition(self, model: ReadOnlyModel):
        """
        Checks that the pipe can be run successfully with the provided model.
        Subclasses can optionally override this method if they wish to perform
        checks before the `run` method. An exception should be thrown if some
        property of the model would not allow running the pipe correctly.
        """

    @abstractmethod
    def run(
        self,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeDependencies:
        """
        Run the pipe with the given model.
        The containers set is the set of ephemeral containers used for this run,
        and they contains both the inputs and outputs of the pipe.
        The incoming requests are the requests that were made to the pipe before
        running it, they are mostly for validation purposes.
        The outgoing requests are the objects that the pipe has to produce in
        the requested containers as a result of running.
        `containers`, `incoming`, and `outgoing` are all lists with the same
        length as `SIGNATURE`.
        """

    def needed_files(self, model: ReadOnlyModel) -> list[FileRequest]:
        """
        Request the list of file hashes that would be requested as part of the
        `run` method via the `FileProvider`. This is required because the list
        of files needs to be known ahead of time in debug mode to dump them on
        disk before running the `run-pipe` command.
        """
        return []

    def requires_custom_invalidation(self, diff: ModelDiff) -> bool:
        """
        Optional method that subclasses can override.
        Given a diff, report if the `invalidate` method should be called with
        the invalidation data to compute additional objects to invalidate.
        """
        return False

    def process_custom_invalidation(
        self, invalidation_data: PipeCustomInvalidation, diff: ModelDiff
    ) -> list[ObjectSet]:
        """
        Optional method that subclasses can override.
        Query the pipe for additional objects to purge, based on the model diff
        and the opaque invalidation data returned by a previous execution of the
        pipe's run method.
        """
        return []

    def has_custom_invalidation(self):
        return (
            self.__class__.requires_custom_invalidation is not Pipe.requires_custom_invalidation
            and self.__class__.process_custom_invalidation is not Pipe.process_custom_invalidation
        )
