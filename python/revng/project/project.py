#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, TypeVar, Union

from revng.project.common import ALL_OBJECTS, AllObjects, ProjectError
from revng.project.model import Binary, BinaryIdentifier, DiffSet  # type: ignore[attr-defined]
from revng.project.model.mixins import AllMixin
from revng.project.pipeline_description import PipelineDescription
from revng.support.artifacts import Artifact
from revng.tupletree import StructBase

StructBaseT = TypeVar("StructBaseT", bound=StructBase)


# TODO: In the future we need to have more features for
#   interactivity/multi-user:
#   1. Cache artifacts
#   2. Handle cache invalidations
#   3. Have some form of locking of the model to enable running
#      "client-side analysis". Without this, complex analysis in Python would
#      be problematic since, without a lock, we might spend a long time doing
#      our computations and then just fail because someone else has renamed a
#      function.
#      The idea would be to have a Python context manager holding the lock.
#   4. We might also have to offer an `async` flavor of the class project.
#      The non-`async` version would have an opt-in thread to monitor
#      notifications coming from the websocket.
class Project(ABC):
    """
    This class is the basis for building a project. It defines
    abstract methods that the inherited classes have to implement
    and it also implements some common methods.
    """

    def __init__(self):
        self.model: Binary = Binary()
        self._last_saved_model: Binary = Binary()
        self._epoch = self._get_epoch()

        self._pipeline_description: PipelineDescription = self._get_pipeline_description()
        self._analysis_names: Set[str] = {
            entry["name"] for entry in self._pipeline_description["analyses"]
        }
        self._analyses_list: Dict[str, List[str]] = {
            entry["name"]: entry["analyses"]
            for entry in self._pipeline_description["analyses_lists"]
        }

        # Compute the container type of an artifact, these will be used later
        # to quickly get the mime type or kind of an artifact
        container_types = {e["class"]: e for e in self._pipeline_description["container_types"]}
        self._artifact_container_types = {}
        for entry in self._pipeline_description["artifacts"]:
            container_type = self._pipeline_description["containers"][entry["container"]]
            self._artifact_container_types[entry["name"]] = container_types[container_type]

    @abstractmethod
    def upload_binary(self, binary_path: Union[str, Path]) -> str:
        """
        Add the specified file to storage. Will return the hash of the file
        with which the file can be referred to.
        """
        pass

    def _get_artifact(
        self,
        artifact_name: str,
        objects: Union[Sequence[Union[str, StructBaseT]], AllObjects] = ALL_OBJECTS,
        configuration: Dict[str, str] = {},
    ) -> Mapping[str, Artifact]:
        """
        Use the method to fetch one or more objects from an artifact.
        `objects` is a list of objects to produce, each element can be either a
        string which is interpreted as a ObjectID string or a model object, which
        will be converted to the corresponding ObjectID string.
        If the `objects` parameter is omitted then all objects of the specified
        artifacts are fetched.
        The return value is `Dict[ObjectID, Artifact]` which maps each object
        to the corresponding artifact.
        """
        actual_objects: Set[str] | AllObjects

        if objects is ALL_OBJECTS:
            actual_objects = ALL_OBJECTS
        else:
            actual_objects = set()
            for object_ in objects:
                if isinstance(object_, str):
                    actual_objects.add(object_)
                elif isinstance(object_, AllMixin):
                    assert object_._location is not None
                    actual_objects.add(object_._location)
                else:
                    raise ValueError(f"Unknown parameter: {object_}")
            if len(actual_objects) == 0:
                return {}

        results = self._get_artifact_impl(artifact_name, actual_objects, configuration)
        container_mime = self._artifact_container_types[artifact_name]["mime_type"]
        return {k: Artifact.make(v, container_mime) for k, v in results.items()}

    @abstractmethod
    def _get_artifact_impl(
        self,
        artifact_name: str,
        objects: Union[Set[str], AllObjects],
        configuration: Dict[str, str],
    ) -> Dict[str, bytes]:
        """
        This method implements the actual fetching of the artifact(s). It is
        implemented by the children of this class.
        """
        pass

    def _analyze(
        self,
        analysis_name: str,
        configuration: Dict[str, str] = {},
        containers: Dict[str, List[str]] = {},
    ):
        """
        Run a single analysis or an analysis list. In addition to the
        `analysis_name` you can optionally specify a dict of objects in
        `containers`. Some analyses require a configuration, which can be
        passed in the `configuration` dict.
        """

        if analysis_name not in self._analyses_list and analysis_name not in self._analysis_names:
            raise ProjectError(f"Analysis {analysis_name} does not exist")
        self._epoch = self._analyze_impl(analysis_name, configuration, containers)

    @abstractmethod
    def _analyze_impl(
        self,
        analysis_name: str,
        configuration: Dict[str, str],
        containers: Dict[str, List[str]],
    ) -> int:
        """
        Actual method that subclasses must implement to run an analysis, it's
        called when preliminary checks have already run, so it's guaranteed
        that all the inputs are well-formed.
        Returns the new epoch after running the analysis (list).
        """
        pass

    def _commit(self):
        diff = DiffSet.make(self._last_saved_model, self.model)
        if len(diff.Changes) == 0:
            return

        self._analyze("apply-diff", configuration={"apply-diff": diff.serialize()})

    def _revert(self):
        """
        Revert changes made to the `self.model` since the last call to `commit()`.
        """
        self._set_model(deepcopy(self._last_saved_model))

    @abstractmethod
    def _get_epoch(self) -> int:
        """
        Get the current epoch at rest. Only done at startup, afterwards the
        epoch will be tracked manually assuming we're the only users.
        """
        pass

    @abstractmethod
    def _get_pipeline_description(self) -> PipelineDescription:
        """
        Define how to implement getting the pipeline description data. This
        method should parse the data and return a PipelineDescription object.
        """
        pass

    def import_and_analyze(self, binary_path: Union[str, Path]):
        """
        A helper method for setting up the project. This method imports
        the binary and runs the initial auto analysis.
        """

        # TODO: remove once we have multi-binary in place
        if len(self.model.Binaries) > 0:
            raise ProjectError(
                "Cannot import a binary when there is one already present in the model"
            )

        # TODO: maybe have a method of that allows checking if the file is
        #       already present
        hash_ = self.upload_binary(binary_path)

        size = Path(binary_path).stat().st_size
        self.model.Binaries.append(
            BinaryIdentifier(Index=0, Hash=hash_, Size=size, CanonicalPath=Path(binary_path).name)
        )
        self._commit()
        self._analyze("initial-auto-analysis")

    def _set_model(self, model: Binary):
        self.model = model
        self._last_saved_model = deepcopy(self.model)
