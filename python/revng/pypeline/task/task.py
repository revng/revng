#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass
from enum import Flag, auto
from typing import Annotated

from revng.pypeline.container import ContainerDeclaration, ContainerID
from revng.pypeline.model import ModelPath
from revng.pypeline.object import ObjectID

PipeObjectDependencies = Annotated[
    list[list[tuple[ObjectID, ModelPath]]],
    """
    A list representing the dependencies between the an object (in a certain container) produced
    by a Pipe. As the Pipe doesn't know the container names, it just returns
    the index of the container in the Pipe's signature. And then it's
    up to `PipelineNode` to remap the index to the container name.
    """,
]

ObjectDependencies = Annotated[
    list[tuple[ContainerID, ObjectID, ModelPath]],
    """
    A list representing the dependencies between the an object (in a certain container) produced
    by a certain task and the model.
    """,
]


class TaskArgumentAccess(Flag):
    READ = auto()
    WRITE = auto()
    READ_WRITE = READ | WRITE


@dataclass(slots=True, frozen=True)
class TaskArgument(ContainerDeclaration):
    """
    A container argument for a task.

    It has a name, it has to be bounded to a ContainerDeclaration and we can specify whether the
    container is just read, just written or both.
    """

    access: TaskArgumentAccess
    # This is the description of the argument that will appear in the CLI
    help_text: str = ""

    def to_container_decl(self) -> ContainerDeclaration:
        """
        Convert this TaskArgument to a ContainerDeclaration.
        """
        return ContainerDeclaration(self.name, self.container_type)
