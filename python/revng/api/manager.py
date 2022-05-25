#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

from ._capi import _api, ffi
from .analysis import Analysis
from .container import Container, ContainerIdentifier
from .exceptions import RevngException
from .kind import Kind
from .step import Step
from .target import Target, TargetsList
from .utils import make_c_string, make_generator, make_python_string, save_file

INVALID_INDEX = 0xFFFFFFFFFFFFFFFF


class Manager:
    def __init__(
        self,
        workdir: str,
    ):
        REVNG_PIPELINES = os.getenv("REVNG_PIPELINES", "")  # noqa: N806
        pipelines = [Path(f) for f in REVNG_PIPELINES.split(",")]
        assert len(pipelines) > 0, "Pipelines must have len > 0"
        assert all(x.is_file() for x in pipelines), "Pipeline files must exist"

        REVNG_FLAGS = os.getenv("REVNG_FLAGS", "")  # noqa: N806
        flags = shlex.split(REVNG_FLAGS)

        _flags = [make_c_string(s) for s in flags]
        _workdir = make_c_string(workdir)
        _pipelines_paths = [make_c_string(str(s.resolve())) for s in pipelines]

        # Ensures that the _manager property is always defined even if the API call fails
        self._manager = None

        self._manager = _api.rp_manager_create(
            len(_pipelines_paths),
            _pipelines_paths,
            len(_flags),
            _flags,
            _workdir,
        )

        assert self._manager, "Failed to instantiate manager"

    def store_containers(self):
        return _api.rp_manager_store_containers(self._manager)

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

    def deserialize_target(self, serialized_target: str, container: Container) -> Target:
        _target = _api.rp_target_create_from_string(self._manager, serialized_target)
        return Target(_target, container)

    def _produce_target(
        self,
        step: Step,
        target: Union[Target, List[Target]],
        container: Container,
    ) -> str:
        if isinstance(target, Target):
            _targets = [target._target]
        else:
            _targets = [t._target for t in target]
        _step = step._step
        _container = container._container
        _product = _api.rp_manager_produce_targets(
            self._manager, len(_targets), _targets, _step, _container
        )
        return make_python_string(_product)

    def produce_target(
        self,
        step_name: str,
        target: Union[None, str, List[str]],
        container_name: Optional[str] = None,
        only_if_ready=False,
    ) -> str:
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
        exact: bool = True,
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

        target = Target.create(kind, container, exact, path_components)
        if target is None:
            raise RevngException("Invalid target")
        return target

    def recalculate_all_available_targets(self):
        _api.rp_manager_recompute_all_available_targets(self._manager)

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

    def get_targets(self, step_name: str, container_name: str, recalc: bool = True) -> List[Target]:
        step = self.get_step(step_name)
        if step is None:
            raise RevngException("Invalid step name")

        container_identifier = self.get_container_with_name(container_name)
        if container_identifier is None:
            raise RevngException("Invalid container name")

        container = step.get_container(container_identifier)
        if container is None:
            raise RevngException(f"Step {step_name} does not use container {container_name}")

        if recalc:
            self.recalculate_all_available_targets()

        targets_list = self.get_targets_list(container)
        if targets_list is None:
            raise RevngException("Invalid container name (cannot get targets list)")

        return list(targets_list.targets())

    def get_targets_from_step(self, step_name: str, recalc: bool = True) -> Dict[str, List[Target]]:
        step = self.get_step(step_name)
        if step is None:
            raise RevngException("Invalid step name")

        if recalc:
            self.recalculate_all_available_targets()

        containers = []
        for container_id in self.containers():
            containers.append(step.get_container(container_id))

        ret = {}
        for container in containers:
            if container is not None:
                targets = self.get_targets(step_name, container.name, False)
                ret[container.name] = targets
        return ret

    def get_all_targets(self) -> Dict[str, Dict[str, List[Target]]]:
        targets: Dict[str, Dict[str, List[Target]]] = {}
        container_ids = list(self.containers())
        self.recalculate_all_available_targets()
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
    ) -> Dict[str, str]:
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

        analysis_result = self._run_analysis(step, analysis, concrete_target_mapping)
        if not analysis_result:
            raise RevngException("Failed to run analysis")
        return analysis_result

    def _run_analysis(
        self, step: Step, analysis: Analysis, target_mapping: Dict[Container, List[Target]]
    ) -> Optional[Dict[str, str]]:
        first_key = list(target_mapping.keys())[0]
        targets = target_mapping[first_key]
        result = _api.rp_manager_run_analysis(
            self._manager,
            len(targets),
            [t._target for t in targets],
            make_c_string(step.name),
            make_c_string(analysis.name),
            first_key._container,
        )
        return self.parse_diff_map(result) if result != ffi.NULL else None

    def run_all_analyses(self) -> Optional[Dict[str, str]]:
        result = _api.rp_manager_run_all_analyses(self._manager)
        return self.parse_diff_map(result) if result != ffi.NULL else None

    def parse_diff_map(self, diff_map) -> Dict[str, str]:
        result = {}
        for global_name in self.globals_list():
            _diff_value = _api.rp_diff_map_get_diff(diff_map, make_c_string(global_name))
            if _diff_value != ffi.NULL:
                result[global_name] = make_python_string(_diff_value)
        return result

    # Global Handling & misc.

    def _get_global(self, name) -> str:
        _name = make_c_string(name)
        _out = _api.rp_manager_create_global_copy(self._manager, _name)
        return make_python_string(_out)

    def _set_global(self, name, content):
        _name = make_c_string(name)
        _content = make_c_string(content)
        _api.rp_manager_set_global(self._manager, _content, _name)

    def get_model(self) -> str:
        return self._get_global("model.yml")

    def globals_count(self) -> int:
        return _api.rp_manager_get_globals_count(self._manager)

    def _get_global_from_index(self, idx: int) -> str:
        _name = _api.rp_manager_get_global_name(self._manager, idx)
        return make_python_string(_name)

    def globals_list(self) -> Generator[str, None, None]:
        return make_generator(self.globals_count(), self._get_global_from_index)

    def set_input(self, container_name: str, content: Union[bytes, str], _key=None) -> str:
        step = self.get_step("begin")
        if step is None:
            return ""

        container_identifier = self.get_container_with_name(container_name)
        if container_identifier is None:
            raise RevngException("Invalid container name")

        container = step.get_container(container_identifier)
        if container is None:
            raise RevngException(f"Step {step.name} does not use container {container_name}")

        # TODO: replace when proper C API is present
        _container_path = self.container_path(step.name, container_name)
        if _container_path is None:
            raise RevngException("Invalid step or container name")

        container_path = Path(_container_path)
        container_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(container_path, content)

        success = container.load(str(container_path))
        if not success:
            raise RevngException(
                f"Failed loading user provided input for container {container_name}"
            )

        return str(container_path.resolve())

    def pipeline_artifact_structure(self):
        structure = defaultdict(list)

        for step in self.steps():
            step_kind = step.get_artifacts_kind()
            if step_kind is not None:
                structure[step_kind.rank].append(step)

        return structure
