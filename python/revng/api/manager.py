#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Generator, Iterable, List, Mapping, Optional, Union

from revng.support import AnyPath

from ._capi import _api, ffi
from .analysis import AnalysesList, Analysis
from .container import Container, ContainerIdentifier
from .errors import Error, Expected
from .exceptions import RevngException
from .invalidations import Invalidations, ResultWithInvalidations
from .kind import Kind
from .step import Step
from .string_map import StringMap
from .target import ContainerToTargetsMap, Target, TargetsList
from .utils import make_c_string, make_generator, make_python_string

INVALID_INDEX = 0xFFFFFFFFFFFFFFFF


class Manager:
    def __init__(
        self,
        workdir: Optional[AnyPath] = None,
        flags: Iterable[str] = (),
    ):
        if workdir is None:
            self.temporary_workdir = TemporaryDirectory(prefix="revng-manager-workdir-")
            self.workdir = Path(self.temporary_workdir.name)
        else:
            self.workdir = Path(workdir)

        self.workdir.mkdir(parents=True, exist_ok=True)

        _flags = [make_c_string(s) for s in flags]
        _workdir = make_c_string(str(self.workdir))

        # Ensures that the _manager property is always defined even if the API call fails
        self._manager = None

        self._manager = _api.rp_manager_create(
            len(_flags),
            _flags,
            _workdir,
        )

        assert self._manager, "Failed to instantiate manager"

    @property
    def uid(self) -> int:
        return int(ffi.cast("uintptr_t", self._manager))

    def save(self):
        return _api.rp_manager_save(self._manager)

    # Kind-related Functions

    @property
    def kinds_count(self) -> int:
        return _api.rp_manager_kinds_count(self._manager)

    def _get_kind_from_index(self, idx: int) -> Optional[Kind]:
        _kind = _api.rp_manager_get_kind(self._manager, idx)
        return Kind(_kind) if _kind != ffi.NULL else None

    def kinds(self) -> Generator[Kind, None, None]:
        return make_generator(self.kinds_count, self._get_kind_from_index)

    def kind_from_name(self, kind_name: str) -> Optional[Kind]:
        _kind_name = make_c_string(kind_name)
        kind = _api.rp_manager_get_kind_from_name(self._manager, _kind_name)
        return Kind(kind) if kind != ffi.NULL else None

    # Target-related Functions

    def deserialize_target(self, serialized_target: str) -> Optional[Target]:
        step_name, container_name, target = serialized_target.split("/", 2)
        target_path, kind_name = target.split(":", 1)
        targets = self.get_targets(step_name, container_name)
        return next((t for t in targets if t.serialize() == target), None)

    def _produce_target(
        self,
        step: Step,
        target: Union[Target, List[Target]],
        container: Container,
    ) -> Dict[str, str | bytes]:
        if isinstance(target, Target):
            targets = [
                target,
            ]
        else:
            targets = target
        _step = step._step
        _container = container._container
        product = _api.rp_manager_produce_targets(
            self._manager, len(targets), [t._target for t in targets], _step, _container
        )
        if not product:
            # TODO: we really should be able to provide a detailed error here
            raise RevngException("Failed to produce targets")

        result = {}
        for target in targets:
            extracted_target = target.extract()
            if extracted_target is None:
                raise RevngException(f"Target {target.serialize()} extraction failed")
            result[target.serialize()] = extracted_target
        return result

    def produce_target(
        self,
        step_name: str,
        target: Union[None, str, List[str]],
        container_name: Optional[str] = None,
        only_if_ready=False,
    ) -> Dict[str, str | bytes]:
        step = self.get_step(step_name)
        if step is None:
            raise RevngException(f"Invalid step {step_name}")

        if container_name is not None:
            container_identifier = self.get_container_with_name(container_name)
            if container_identifier is None:
                raise RevngException(f"Invalid container {container_name}")

            container = step.get_container(container_identifier)
            if container is None:
                raise RevngException(f"Step {step_name} does not use container {container_name}")
        else:
            container = step.get_artifacts_container()
            if container is None:
                raise RevngException(f"Step {step_name} does not have an artifacts container")

        if target is None:
            _targets = [
                "",
            ]
        elif isinstance(target, str):
            _targets = [
                target,
            ]
        else:
            _targets = target

        targets: List[Target] = []
        for _target_elem in _targets:
            if container_name is not None:
                targets.append(self.create_target(_target_elem, container))
            else:
                targets.append(self.create_target(_target_elem, container, step))

        if only_if_ready and any(not t.is_ready for t in targets):
            raise RevngException("Requested production of unready targets")

        product = self._produce_target(step, targets, container)
        if not product:
            # TODO: we really should be able to provide a detailed error here
            raise RevngException("Failed to produce target")
        return product

    def create_target(
        self,
        target_path: str,
        container: Container,
        step: Optional[Step] = None,
    ) -> Target:
        if step is not None:
            path = target_path
            kind = step.get_artifacts_kind()
            if kind is None:
                raise RevngException("Step does not have an artifacts kind")
        else:
            components = target_path.split(":")
            path, kind_name = ":".join(components[:-1]), components[-1]

            kind = self.kind_from_name(kind_name)
            if kind is None:
                raise RevngException("Invalid kind")

        path_components = path.split("/") if path != "" else []

        if kind.rank is not None and len(path_components) != kind.rank.depth:
            raise RevngException("Path components need to equal kind rank")

        target = Target.create(kind, container, path_components)
        if target is None:
            raise RevngException("Invalid target")
        return target

    def get_targets_list(self, container: Container) -> Optional[TargetsList]:
        targets_list = _api.rp_manager_get_container_targets_list(
            self._manager, container._container
        )

        return TargetsList(targets_list, container) if targets_list != ffi.NULL else None

    # Container-related functions

    def container_path(self, step_name: str, container_name: str) -> Optional[str]:
        _step_name = make_c_string(step_name)
        _container_name = make_c_string(container_name)
        _path = _api.rp_manager_create_container_path(self._manager, _step_name, _container_name)
        if not _path:
            return None
        return make_python_string(_path)

    @property
    def containers_count(self) -> int:
        return _api.rp_manager_containers_count(self._manager)

    def containers(self) -> Generator[ContainerIdentifier, None, None]:
        return make_generator(self.containers_count, self._get_container_identifier)

    def get_container_with_name(self, name) -> Optional[ContainerIdentifier]:
        for container in self.containers():
            if container.name == name:
                return container
        return None

    def deserialize_container(self, step: Step, container_name: str, content: bytes):
        _content = ffi.from_buffer(content)
        return _api.rp_manager_container_deserialize(
            self._manager, step._step, make_c_string(container_name), _content, len(_content)
        )

    def _get_container_identifier(self, idx: int) -> Optional[ContainerIdentifier]:
        _container_identifier = _api.rp_manager_get_container_identifier(self._manager, idx)
        if _container_identifier != ffi.NULL:
            return ContainerIdentifier(_container_identifier)
        return None

    # Step-related functions

    @property
    def steps_count(self) -> int:
        return _api.rp_manager_steps_count(self._manager)

    def get_step(self, step_name: str) -> Optional[Step]:
        step_index = self._step_name_to_index(step_name)
        if step_index is None:
            return None
        return self._get_step_from_index(step_index)

    def steps(self) -> Generator[Step, None, None]:
        return make_generator(self.steps_count, self._get_step_from_index)

    def _step_name_to_index(self, name: str) -> Optional[int]:
        _name = make_c_string(name)
        index = _api.rp_manager_step_name_to_index(self._manager, _name)
        return index if index != INVALID_INDEX else None

    def _get_step_from_index(self, idx: int) -> Optional[Step]:
        step = _api.rp_manager_get_step(self._manager, idx)
        return Step(step) if step != ffi.NULL else None

    # Utility target functions

    def get_targets(self, step_name: str, container_name: str) -> List[Target]:
        step = self.get_step(step_name)
        if step is None:
            raise RevngException("Invalid step name")

        container_identifier = self.get_container_with_name(container_name)
        if container_identifier is None:
            raise RevngException("Invalid container name")

        container = step.get_container(container_identifier)
        if container is None:
            raise RevngException(f"Step {step_name} does not use container {container_name}")

        targets_list = self.get_targets_list(container)
        return list(targets_list.targets()) if targets_list is not None else []

    def get_targets_from_step(self, step_name: str) -> Dict[str, List[Target]]:
        step = self.get_step(step_name)
        if step is None:
            raise RevngException("Invalid step name")

        containers = []
        for container_id in self.containers():
            containers.append(step.get_container(container_id))

        ret = {}
        for container in containers:
            if container is not None:
                targets = self.get_targets(step_name, container.name)
                ret[container.name] = targets
        return ret

    def get_all_targets(self) -> Dict[str, Dict[str, List[Target]]]:
        targets: Dict[str, Dict[str, List[Target]]] = {}
        container_ids = list(self.containers())
        for step in self.steps():
            targets[step.name] = {}
            containers = [step.get_container(cid) for cid in container_ids]
            for container in [c for c in containers if c is not None]:
                target_list = self.get_targets_list(container)
                if target_list is not None:
                    target_dicts = list(target_list.targets())
                    targets[step.name][container.name] = target_dicts
        return targets

    # Analysis handling

    def run_analysis(
        self,
        step_name: str,
        analysis_name: str,
        target_mapping: Dict[str, List[str]],
        options: Dict[str, str] | None = None,
    ) -> ResultWithInvalidations[Dict[str, str]]:
        step = self.get_step(step_name)
        if step is None:
            raise RevngException(f"Invalid step {step_name}")

        analysis = next((a for a in step.analyses() if a.name == analysis_name), None)
        if analysis is None:
            raise RevngException(f"Invalid Analysis {analysis_name}")

        concrete_target_mapping = {}
        for container_name, target_list in target_mapping.items():
            container_identifier = self.get_container_with_name(container_name)
            if container_identifier is None:
                raise RevngException(f"Invalid container {container_name}")

            container = step.get_container(container_identifier)
            if container is None:
                raise RevngException(f"Step {step_name} does not use container {container_name}")

            concrete_targets = [
                self.create_target(target, container, None) for target in target_list
            ]

            analysis_argument = next(a for a in analysis.arguments() if a.name == container_name)
            for target in concrete_targets:
                if target.kind is None or target.kind.name not in [
                    k.name for k in analysis_argument.acceptable_kinds()
                ]:
                    raise RevngException(
                        f"Wrong kind for analysis: found '{target.kind}', "
                        + f"expected: {[a.name for a in analysis_argument.acceptable_kinds()]}"
                    )
            concrete_target_mapping[container] = concrete_targets

        options_map = StringMap(options)
        analysis_result = self._run_analysis(step, analysis, concrete_target_mapping, options_map)
        if analysis_result.result is None:
            raise RevngException("Failed to run analysis")
        return analysis_result  # type: ignore

    def _run_analysis(
        self,
        step: Step,
        analysis: Analysis,
        target_mapping: Dict[Container, List[Target]],
        options: StringMap,
    ) -> ResultWithInvalidations[Optional[Dict[str, str]]]:
        target_map = ContainerToTargetsMap()
        for container, targets in target_mapping.items():
            target_map.add(container, *targets)

        invalidations = Invalidations()
        result = _api.rp_manager_run_analysis(
            self._manager,
            make_c_string(step.name),
            make_c_string(analysis.name),
            target_map._map,
            invalidations._invalidations,
            options._string_map,
        )

        if result != ffi.NULL and not _api.rp_diff_map_is_empty(result):
            self.save()
        return ResultWithInvalidations(
            self.parse_diff_map(result) if result != ffi.NULL else None, invalidations
        )

    def run_analyses_list(
        self, analyses_list: AnalysesList | str, options: Mapping[str, str] | None = None
    ) -> ResultWithInvalidations[Optional[Dict[str, str]]]:
        if isinstance(analyses_list, str):
            name = analyses_list
            found_al = next((al for al in self.analyses_lists() if al.name == name), None)
            if found_al is not None:
                real_analyses_list = found_al
            else:
                raise RevngException(f"Could not find analyses list {name}")
        else:
            real_analyses_list = analyses_list

        options_map = StringMap(options)
        invalidations = Invalidations()
        result = _api.rp_manager_run_analyses_list(
            self._manager,
            real_analyses_list._analyses_list,
            invalidations._invalidations,
            options_map._string_map,
        )
        if result != ffi.NULL and not _api.rp_diff_map_is_empty(result):
            self.save()
        return ResultWithInvalidations(
            self.parse_diff_map(result) if result != ffi.NULL else None, invalidations
        )

    def parse_diff_map(self, diff_map) -> Dict[str, str]:
        result = {}
        for global_name in self.globals_list():
            _diff_value = _api.rp_diff_map_get_diff(diff_map, make_c_string(global_name))
            if _diff_value != ffi.NULL:
                result[global_name] = make_python_string(_diff_value)
        return result

    def _analyses_list_count(self) -> int:
        return _api.rp_manager_get_analyses_list_count(self._manager)

    def _analyses_list_get(self, index: int) -> AnalysesList:
        _analyses_list = _api.rp_manager_get_analyses_list(self._manager, index)
        return AnalysesList(_analyses_list, self)

    def analyses_lists(self) -> Generator[AnalysesList, None, None]:
        return make_generator(self._analyses_list_count(), self._analyses_list_get)

    # Global Handling & misc.

    def get_global(self, name) -> str:
        _name = make_c_string(name)
        _out = _api.rp_manager_create_global_copy(self._manager, _name)
        return make_python_string(_out)

    def set_global(self, name, content) -> ResultWithInvalidations[Expected[bool]]:
        _name = make_c_string(name)
        _content = make_c_string(content)
        invalidations = Invalidations()
        error = Error()
        res = _api.rp_manager_set_global(
            self._manager, _content, _name, invalidations._invalidations, error._error
        )
        return ResultWithInvalidations(Expected(res, error), invalidations)

    def verify_global(self, name, content) -> Expected[bool]:
        _name = make_c_string(name)
        _content = make_c_string(content)
        error = Error()
        res = _api.rp_manager_verify_global(self._manager, _content, _name, error._error)
        return Expected(res, error)

    def apply_diff(self, name, diff) -> ResultWithInvalidations[Expected[bool]]:
        _name = make_c_string(name)
        _diff = make_c_string(diff)
        invalidations = Invalidations()
        error = Error()
        res = _api.rp_manager_apply_diff(
            self._manager, _diff, _name, invalidations._invalidations, error._error
        )
        return ResultWithInvalidations(Expected(res, error), invalidations)

    def verify_diff(self, name, diff) -> Expected[bool]:
        _name = make_c_string(name)
        _diff = make_c_string(diff)
        error = Error()
        res = _api.rp_manager_verify_diff(self._manager, _diff, _name, error._error)
        return Expected(res, error)

    def get_model(self) -> str:
        return self.get_global("model.yml")

    def globals_count(self) -> int:
        return _api.rp_manager_get_globals_count(self._manager)

    def _get_global_from_index(self, idx: int) -> str:
        _name = _api.rp_manager_get_global_name(self._manager, idx)
        return make_python_string(_name)

    def globals_list(self) -> Generator[str, None, None]:
        return make_generator(self.globals_count(), self._get_global_from_index)

    def set_input(self, container_name: str, content: bytes, _key=None):
        step = self.get_step("begin")
        if step is None:
            raise RevngException('Step "begin" not found')

        success = self.deserialize_container(step, container_name, content)

        if not success:
            raise RevngException(
                f"Failed loading user provided input for container {container_name}"
            )
