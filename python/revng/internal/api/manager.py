#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from tempfile import TemporaryDirectory
from threading import Lock
from typing import Dict, Iterable, List, Mapping, Optional

import yaml

from revng.pipeline_description import YamlLoader  # type: ignore

from ._capi import _api, ffi
from .errors import Error, Expected
from .exceptions import RevngException, RevngManagerInstantiationException
from .invalidations import Invalidations, ResultWithInvalidations
from .string_map import StringMap
from .target import ContainerToTargetsMap, Target, TargetsList
from .utils import make_c_string, make_python_string


class Manager:
    def __init__(
        self,
        workdir: Optional[str] = None,
        flags: Iterable[str] = (),
    ):
        if workdir is None:
            self.temporary_workdir = TemporaryDirectory(prefix="revng-manager-workdir-")
            self.workdir = self.temporary_workdir.name
        else:
            self.workdir = workdir

        _flags = [make_c_string(s) for s in flags]
        _workdir = make_c_string(str(self.workdir))

        # Initialize the save lock, this needs to be acquired when either
        # saving or running analyses. This is only needed because the workdir
        # needs to be moved to a temporary directory before it is safe for
        # another thread to save again
        self.save_lock = Lock()

        # Ensures that the _manager property is always defined even if the API call fails
        self._manager = None

        self._manager = _api.rp_manager_create(
            len(_flags),
            _flags,
            _workdir,
        )

        if self._manager in (None, ffi.NULL):
            raise RevngManagerInstantiationException("Failed to instantiate manager")

        _description = _api.rp_manager_get_pipeline_description(self._manager)
        self._description_str = make_python_string(_description)
        self.description = yaml.load(self._description_str, Loader=YamlLoader)

    @property
    def uid(self) -> int:
        return int(ffi.cast("uintptr_t", self._manager))

    def save(self) -> bool:
        return _api.rp_manager_save(self._manager)

    # description utilities

    def kind_from_name(self, name: str):
        return next((k for k in self.description.Kinds if k.Name == name), None)

    def rank_from_name(self, name: str):
        return next((r for r in self.description.Ranks if r.Name == name), None)

    def analysis_from_name(self, step: str, name: str):
        step_obj = self.step_from_name(step)
        if step_obj is None:
            return None

        return next((a for a in step_obj.Analyses if a.Name == name), None)

    def analyses_list_from_name(self, name: str):
        return next((al for al in self.description.AnalysesLists if al.Name == name), None)

    def step_from_name(self, name: str):
        return next((s for s in self.description.Steps if s.Name == name), None)

    def container_from_name(self, name: str):
        return next((c for c in self.description.Containers if c.Name == name), None)

    # Pointer getters

    def _get_step_ptr(self, step_name: str):
        if self.step_from_name(step_name) is None:
            raise RevngException(f"Invalid step {step_name}")

        return _api.rp_manager_get_step_from_name(self._manager, make_c_string(step_name))

    def _get_container_identifier_ptr(self, container_name: str):
        if self.container_from_name(container_name) is None:
            raise RevngException(f"Invalid container {container_name}")

        return _api.rp_manager_get_container_identifier_from_name(
            self._manager, make_c_string(container_name)
        )

    def _get_container_ptr_from_step(self, step_name, step_ptr, container_name: str):
        identifier_ptr = self._get_container_identifier_ptr(container_name)

        container_ptr = _api.rp_step_get_container(step_ptr, identifier_ptr)
        if container_ptr == ffi.NULL:
            raise RevngException(f"Step {step_name} does not use container {container_name}")

        return step_ptr, container_ptr

    def _get_step_container_ptr(self, step_name: str, container_name: str):
        step_ptr = self._get_step_ptr(step_name)
        return self._get_container_ptr_from_step(step_name, step_ptr, container_name)

    # Target-related Functions

    def deserialize_target(self, serialized_target: str) -> Optional[Target]:
        step_name, container_name, target = serialized_target.split("/", 2)
        target_path, kind_name = target.split(":", 1)
        targets = self.get_targets(step_name, container_name)
        return next((t for t in targets if t.serialize() == target), None)

    def produce_target(
        self,
        step_name: str,
        target: None | str | List[str],
        container_name: Optional[str] = None,
        only_if_ready=False,
    ) -> Dict[str, str | bytes] | Error:
        step = self.step_from_name(step_name)
        if step is None:
            raise RevngException(f"Invalid step {step_name}")

        if container_name is not None:
            container = container_name
        else:
            container = step.Artifacts.Container
            if container == "":
                raise RevngException(f"Step {step_name} does not have an artifacts container")

        if target is None:
            _targets = [""]
        elif isinstance(target, str):
            _targets = [target]
        else:
            _targets = target

        targets: List[Target] = []
        for _target_elem in _targets:
            targets.append(
                self.create_target(step_name, container, _target_elem, container_name is None)
            )

        if only_if_ready and any(not t.is_ready for t in targets):
            raise RevngException("Requested production of unready targets")

        _step, _container = self._get_step_container_ptr(step_name, container)
        error = Error()
        product = _api.rp_manager_produce_targets(
            self._manager,
            _step,
            _container,
            len(targets),
            [t._target for t in targets],
            error._error,
        )

        if product == ffi.NULL:
            return error

        result = {}
        for produced_target in targets:
            extracted_target = produced_target.extract()
            if extracted_target is None:
                raise RevngException(f"Target {produced_target.serialize()} extraction failed")
            result[produced_target.serialize()] = extracted_target
        return result

    def create_target(
        self, step_name: str, container_name: str, target_path: str, use_artifact_kind: bool
    ) -> Target:
        step_ptr, container_ptr = self._get_step_container_ptr(step_name, container_name)
        step_obj = self.step_from_name(step_name)
        assert step_obj is not None

        if use_artifact_kind:
            if step_obj.Artifacts.Kind == "":
                raise RevngException("Step does not have an artifacts kind")

            kind_name = step_obj.Artifacts.Kind
            kind = self.kind_from_name(kind_name)
            assert kind is not None

            path_components = target_path.split("/") if target_path != "" else []
        else:
            path, kind_name = target_path.rsplit(":", 1)
            kind = self.kind_from_name(kind_name)
            if kind is None:
                raise RevngException(f"Kind {kind_name} does not exist")

            path_components = path.split("/") if path != "" else []

        rank = self.rank_from_name(kind.Rank)  # type: ignore
        if rank is not None and len(path_components) != rank.Depth:
            raise RevngException("Path components need to equal kind rank")

        _kind = _api.rp_manager_get_kind_from_name(self._manager, make_c_string(kind_name))
        assert _kind != ffi.NULL

        target = Target.create(_kind, container_ptr, path_components)
        if target is None:
            raise RevngException("Invalid target")

        return target

    # Utility target functions

    def get_targets(self, step_name: str, container_name: str) -> List[Target]:
        _, container_ptr = self._get_step_container_ptr(step_name, container_name)

        targets_list = _api.rp_manager_get_container_targets_list(self._manager, container_ptr)

        if targets_list != ffi.NULL:
            return list(TargetsList(targets_list, container_ptr))
        else:
            return []

    # Analysis handling

    def run_analysis(
        self,
        step_name: str,
        analysis_name: str,
        target_mapping: Dict[str, List[str]],
        options: Dict[str, str] | None = None,
    ) -> Expected[ResultWithInvalidations[Dict[str, str]]]:
        step = self.step_from_name(step_name)
        if step is None:
            raise RevngException(f"Invalid step {step_name}")

        analysis = self.analysis_from_name(step_name, analysis_name)
        if analysis is None:
            raise RevngException(f"Invalid Analysis {analysis_name}")

        concrete_target_mapping = {}
        for container_name, target_list in target_mapping.items():
            concrete_targets = [
                self.create_target(step_name, container_name, target, False)
                for target in target_list
            ]

            analysis_ci = next(ci for ci in analysis.ContainerInputs if ci.Name == container_name)
            for target in concrete_targets:
                if target.kind is None or target.kind not in analysis_ci.AcceptableKinds:
                    raise RevngException(
                        f"Wrong kind for analysis: found '{target.kind}', "
                        + f"expected: {[analysis_ci.AcceptableKinds]}"
                    )
            concrete_target_mapping[container_name] = concrete_targets

        options_map = StringMap(options)
        analysis_result = self._run_analysis(
            step_name, analysis_name, concrete_target_mapping, options_map
        )
        return analysis_result

    def _run_analysis(
        self,
        step_name: str,
        analysis_name: str,
        target_mapping: Dict[str, List[Target]],
        options: StringMap,
    ) -> Expected[ResultWithInvalidations[Dict[str, str]]]:
        target_map = ContainerToTargetsMap()
        step_ptr = self._get_step_ptr(step_name)

        for container, targets in target_mapping.items():
            _, container_ptr = self._get_container_ptr_from_step(step_name, step_ptr, container)
            for target in targets:
                target_map.add(container_ptr, target)

        invalidations = Invalidations()
        error = Error()
        result = _api.rp_manager_run_analysis(
            self._manager,
            make_c_string(step_name),
            make_c_string(analysis_name),
            target_map._map,
            options._string_map,
            invalidations._invalidations,
            error._error,
        )

        return Expected(
            ResultWithInvalidations(
                self.parse_diff_map(result) if result != ffi.NULL else None, invalidations
            ),
            error,
        )

    def run_analyses_list(
        self, analyses_list_name: str, options: Mapping[str, str] | None = None
    ) -> Expected[ResultWithInvalidations[Dict[str, str]]]:
        if self.analyses_list_from_name(analyses_list_name) is None:
            raise RevngException(f"Could not find analyses list {analyses_list_name}")

        options_map = StringMap(options)
        invalidations = Invalidations()
        error = Error()
        result = _api.rp_manager_run_analyses_list(
            self._manager,
            make_c_string(analyses_list_name),
            options_map._string_map,
            invalidations._invalidations,
            error._error,
        )

        return Expected(
            ResultWithInvalidations(
                self.parse_diff_map(result) if result != ffi.NULL else None, invalidations
            ),
            error,
        )

    def parse_diff_map(self, diff_map) -> Dict[str, str]:
        result = {}
        for global_name in self.description.Globals:
            _diff_value = _api.rp_diff_map_get_diff(diff_map, make_c_string(global_name))
            if _diff_value != ffi.NULL:
                result[global_name] = make_python_string(_diff_value)
        return result

    # Global Handling & misc.

    def get_global(self, name) -> str:
        _name = make_c_string(name)
        _out = _api.rp_manager_create_global_copy(self._manager, _name)
        return make_python_string(_out)

    def set_input(self, container_name: str, content: bytes, _key=None) -> Invalidations:
        step_ptr = self._get_step_ptr("begin")

        _content = ffi.from_buffer(content)
        invalidations = Invalidations()
        success = _api.rp_manager_container_deserialize(
            self._manager,
            step_ptr,
            make_c_string(container_name),
            _content,
            len(_content),
            invalidations._invalidations,
        )

        if not success:
            raise RevngException(
                f"Failed loading user provided input for container {container_name}"
            )

        return invalidations

    def get_pipeline_description(self):
        return self._description_str

    def set_storage_credentials(self, credentials: str):
        pass

    def get_context_commit_index(self) -> int:
        return _api.rp_manager_get_context_commit_index(self._manager)
