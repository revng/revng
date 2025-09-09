#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass

from .container import Container, ContainerDeclaration
from .model import Model
from .object import ObjectSet
from .pipeline_node import PipelineNode
from .utils.cabc import ABC, abstractmethod


class Analysis(ABC):
    """
    An analysis makes changes to the model. In order to do this, it might inspect
    previously produced results of the pipeline.

    An analysis is the way in which users are expected to make changes to the model.
    The changes applied by a run of an analysis might lead to invalidate certain
    objects in save points.
    """

    __slots__: tuple = ("name",)

    def __init__(self, name: str):
        """
        Initialize the analysis with a name.
        The name is used for debugging and logging purposes.
        """
        self.name = name

    @classmethod
    @abstractmethod
    def signature(cls) -> tuple[type[Container], ...]:
        """
        The containers required by the analysis, it needs to be a class property
        to auto-generate the CLI commands.
        """
        raise NotImplementedError()

    @abstractmethod
    def run(
        self,
        model: Model,
        containers: list[Container],
        incoming: list[ObjectSet],
        configuration: str,
    ):
        """
        Run the analysis on the model, using the containers and
        incoming requests. The analysis will modify inplace the
        model so you must make a copy before running it, so you
        can compute the diff for invalidation purposes.
        """
        raise NotImplementedError()


@dataclass(frozen=True, slots=True)
class AnalysisBinding:
    """Allows to bind an analysis to a pipeline node."""

    analysis: Analysis
    bindings: tuple[ContainerDeclaration, ...]
    node: PipelineNode
