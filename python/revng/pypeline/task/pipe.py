#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Optional, final

from revng.pypeline.container import Configuration, Container
from revng.pypeline.model import ReadOnlyModel
from revng.pypeline.object import ObjectSet
from revng.pypeline.storage.file_provider import FileProvider
from revng.pypeline.utils.cabc import ABC, abstractmethod

from .task import PipeObjectDependencies, TaskArgument, TaskArgumentAccess


class Pipe(ABC):
    """
    A Pipe is a task that, given some input objects, a configuration string and the model, produces
    some new objects.
    """

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
        self.name = self.__class__.__name__
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

    @abstractmethod
    def run(
        self,
        file_provider: FileProvider,
        model: ReadOnlyModel,
        containers: list[Container],
        incoming: list[ObjectSet],
        outgoing: list[ObjectSet],
        configuration: Configuration,
    ) -> PipeObjectDependencies:
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
