#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass
from enum import Flag, auto

from revng.pypeline.container import ContainerDeclaration


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

    def declaration(self) -> ContainerDeclaration:
        """
        Convert this TaskArgument to a ContainerDeclaration.
        """
        return ContainerDeclaration(self.name, self.container_type)
