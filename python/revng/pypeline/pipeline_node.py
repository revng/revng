#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Annotated, List, Mapping, Optional, Sequence, Set, Union
from typing import overload

from revng.pypeline.analysis import Analysis

from .container import Configuration, ConfigurationId, Container, ContainerDeclaration
from .model import ReadOnlyModel
from .object import ObjectSet
from .storage.file_provider import FileProvider
from .storage.storage_provider import SavePointsRange, StorageProvider
from .task.pipe import Pipe, PipeDependencies
from .task.requests import Requests
from .task.savepoint import SavePoint
from .task.task import TaskArgument

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison


PipelineConfiguration = Annotated[
    Mapping[Pipe | Analysis, str],
    "Configuration for an execution of a schedule or analysis, this describes "
    "the dynamic configuration of each component of the pipeline",
]


def deterministic_sort_key(node: PipelineNode) -> SupportsRichComparison:
    """
    A deterministic sort key for a PipelineNode, used to ensure that the
    savepoint ranges are assigned in a deterministic order.
    """
    return (
        # Arguments of the task
        tuple(node.argument_declarations),
        # Sorted list of outdegree of successors to try to embed the shape
        # of the pipeline in the sort key
        tuple(sorted(len(succ.successors) for succ in node.successors)),
        node.task.name,
    )


class PipelineNode:
    """
    A node of the tree of elements composing a pipeline.

    Mostly a wrapper around a Task.
    """

    __slots__ = (
        "successors",
        "predecessors",
        "task",
        "bindings",
        "reverse_bindings",
        "pipe_dependencies",
        "savepoint_range",
        "id",
    )

    @overload
    def __init__(self, task: Pipe, bindings: Sequence[ContainerDeclaration]): ...

    @overload
    def __init__(self, task: SavePoint): ...

    def __init__(
        self,
        task: Union[Pipe, SavePoint],
        bindings: Sequence[ContainerDeclaration] | None = None,
    ):
        """
        The bindings are used to map between the pipeline declarations and the
        task arguments and are needed only for Pipe tasks as SavePoints only
        live in the pipeline namespace.
        """
        self.successors: List[PipelineNode] = []
        self.predecessors: List[PipelineNode] = []
        self.task = task
        self.pipe_dependencies: Set[Pipe] = set()
        self.bindings: list[ContainerDeclaration] = []
        self.savepoint_range: Optional[SavePointsRange] = None
        self.id = -1

        if bindings is None:
            assert isinstance(
                task, SavePoint
            ), f"A pipeline node with no bindings can only be a Savepoint, got {task}"
            self.bindings = []
        else:
            assert isinstance(task, Pipe), f"Bindings are only allowed for Pipe tasks, got {task}"
            if len(bindings) != len(task.arguments):
                raise ValueError(
                    f"Expected {len(task.arguments)} bindings but got "
                    f"{len(bindings)} for task {task}. arguments: "
                    f"{task.arguments}, bindings: {bindings}"
                )
            for bind, arg in zip(bindings, task.arguments):
                if bind.container_type != arg.container_type:
                    raise TypeError(
                        f"Binding {bind} is not compatible with argument {arg} " f"for task {task}"
                    )
            # These maps the declarations of the pipeline to the task arguments declarations
            self.bindings = list(bindings)
            assert len(self.bindings) == len(task.arguments)

    @property
    def arguments(self) -> list[TaskArgument]:
        if isinstance(self.task, SavePoint):
            # SavePoints do not have bindings, so we return the task arguments directly
            return self.task.arguments
        elif isinstance(self.task, Pipe):
            # For Pipes, we return the bindings, which are the pipeline declarations
            # that map to the task arguments
            result = []
            for index, argument in enumerate(self.bindings):
                pipe_arguments = self.task.arguments[index]
                result.append(
                    TaskArgument(argument.name, argument.container_type, pipe_arguments.access)
                )
            return result
        else:
            raise TypeError(f"Unsupported task type: {type(self.task)}")

    @property
    def argument_declarations(self) -> list[ContainerDeclaration]:
        return [x.declaration() for x in self.arguments]

    def add_successor(self, node: PipelineNode) -> PipelineNode:
        node.predecessors.append(self)
        self.successors.append(node)
        return node

    def sorted_successors(self) -> list[PipelineNode]:
        """
        Return the successors of this node sorted by their deterministic sort key.
        This is needed to ensure that the savepoint ranges are assigned in a
        deterministic order, otherwise the caches could cause problems.
        """
        return sorted(self.successors, key=deterministic_sort_key)

    def prerequisites_for(
        self,
        model: ReadOnlyModel,
        requests: Requests,
        configuration: PipelineConfiguration,
        storage_provider: StorageProvider,
    ) -> Requests:
        # In order to execute the task we might need to remap the requests
        # from the pipeline declarations to the task arguments declarations.
        # So we apply a remap before and after
        if isinstance(self.task, SavePoint):
            # SavePoints have a different semantics for prerequisites, so we call the method
            # from the SavePoint class
            assert (
                self.savepoint_range is not None
            ), "SavePoint range must be set before calling prerequisites_for on a SavePoint"
            return self.task.prerequisites_for(
                requests=requests,
                configuration_id=self.configuration_id(configuration),
                storage_provider=storage_provider,
                savepoint_range=self.savepoint_range,
            )
        elif isinstance(self.task, Pipe):
            # Use the bindings to map the requests to the pipe arguments
            pipe_requests: list[ObjectSet] = [requests.get(decl) for decl in self.bindings]
            # Ask the pipe for its prerequisites
            outgoing = self.task.prerequisites_for(
                model=model,
                requests=pipe_requests,
            )
            # Map back the requests to the pipeline declarations
            result = requests.clone()
            for decl, request in zip(self.bindings, outgoing):
                result[decl] = request
            return result
        else:
            raise TypeError(f"Unsupported task type: {type(self.task)}")

    def configuration_id(self, configuration: PipelineConfiguration) -> ConfigurationId:
        hasher = sha256()
        if isinstance(self.task, Pipe) and self.task in configuration:
            hasher.update(configuration[self.task].encode())
        elif isinstance(self.task, SavePoint):
            for pipe in sorted(self.pipe_dependencies, key=lambda x: (x.name, x.signature())):
                hasher.update(configuration.get(pipe, "").encode())
        return hasher.hexdigest()

    def __repr__(self):
        return f"<PipelineNode: {self.task!r}>"

    def to_dict(self) -> dict:
        """Convert the data into a dictionary representation."""
        return {
            "id": self.id,
            "type": "pipe" if isinstance(self.task, Pipe) else "savepoint",
            "class": self.task.__class__.__name__,
            "arguments": [binding.name for binding in self.bindings],
            "successors": [succ.id for succ in self.successors],
            "predecessors": [pred.id for pred in self.predecessors],
            "savepoint_range": (
                {
                    "start": self.savepoint_range.start,
                    "end": self.savepoint_range.end,
                }
                if self.savepoint_range
                else None
            ),
        }


class DummyPipelineNode(PipelineNode):
    """Dummy pipeline node, to be used in cases where multiple PipelineNode
    branches need to be merged into one"""

    class DummyPipe(Pipe):
        name = "dummy-pipe"

        def __init__(self, static_configuration: str):
            self.static_configuration = ""

        @classmethod
        def signature(cls) -> tuple[TaskArgument, ...]:
            return ()

        def run(
            self,
            file_provider: FileProvider,
            model: ReadOnlyModel,
            containers: list[Container],
            incoming: list[ObjectSet],
            outgoing: list[ObjectSet],
            configuration: Configuration,
        ) -> PipeDependencies:
            return PipeDependencies([])

    def __init__(self):
        # Create a new class object of DummyPipe, with the `name` class
        # attribute changed with the provided one
        super().__init__(self.__class__.DummyPipe(""), [])
