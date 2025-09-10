#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# It's important that the `cli` submodule is NOT imported here
# as for the cli to work we need to first load the pipebox and then
# load the cli submodules so they can access the custom types

__version__ = "@VERSION@"

from .analysis import Analysis
from .container import Container
from .model import Model
from .object import Kind, ObjectID
from .task.pipe import Pipe
from .utils.registry import get_singleton, register_all_subclasses


def initialize_pypeline() -> None:
    """
    This function is used to initialize the pypeline module.
    It is has to be called just after importing the pipebox.

    All of this can be done more robustly using `__init_subclass__`,
    but it's not well supported by nanobind, so we are forced to do it
    manually here.
    """
    register_all_subclasses(Analysis)
    register_all_subclasses(Pipe)
    register_all_subclasses(Container)
    register_all_subclasses(Model, singleton=True)
    register_all_subclasses(Kind, singleton=True)
    register_all_subclasses(ObjectID, singleton=True)
    kind_ty = get_singleton(Kind)  # type: ignore[type-abstract]
    kind_ty._init_type()
