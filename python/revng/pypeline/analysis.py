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

    name: str

    def __init__(self):
        pass

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

    def is_available(self) -> bool:
        """
        Asks if the analysis is available for running. Running an analysis that
        is not available for running will result in an error.
        """
        return True


@dataclass(frozen=True, slots=True)
class AnalysisBinding:
    """Allows to bind an analysis to a pipeline node."""

    analysis: Analysis
    bindings: tuple[ContainerDeclaration, ...]
    node: PipelineNode

    def to_dict(self) -> dict:
        """Convert the data into a dictionary representation."""
        return {
            "name": self.analysis.name,
            "is_available": self.analysis.is_available(),
            "bindings": [
                {
                    "name": binding.name,
                    "container_type": binding.container_type.name,
                }
                for binding in self.bindings
            ],
            "node": self.node.id,
        }


@dataclass(frozen=True, slots=True)
class AnalysisList:
    """A named list of analysis to run in sequence."""

    name: str
    analyses: list[str]
    description: str | None = None

    def __post_init__(self):
        if len(self.analyses) == 0:
            raise ValueError("An analysis list must contain at least one analysis.")

        if len(set(self.analyses)) != len(self.analyses):
            raise ValueError("An analysis list cannot contain duplicate analyses.")

    def to_dict(self) -> dict:
        """Convert the data into a dictionary representation."""
        return {
            "name": self.name,
            "analyses": self.analyses,
            "description": self.description,
        }
