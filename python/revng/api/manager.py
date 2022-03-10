#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from pathlib import Path
from typing import List, Optional, Union, Generator, Dict, cast

from ._capi import _api, ffi
from .container import ContainerIdentifier, Container
from .exceptions import RevngException
from .kind import Kind
from .step import Step
from .target import Target, TargetsList
from .utils import make_c_string, make_python_string, make_generator, save_file

INVALID_INDEX = 0xFFFFFFFFFFFFFFFF


class Manager:
    def __init__(
        self,
        flags: List[str],
        workdir: str,
        pipelines: Union[List[str], List[Path]],
    ):
        _flags = [make_c_string(s) for s in flags]
        _workdir = make_c_string(workdir)

        # Ensures that the _manager property is always defined even if the API
        # call fails
        self._manager = None

        if len(pipelines) == 0:
            assert False, "Pipelines must have len > 0"
        elif isinstance(pipelines[0], Path):
            pipelines = cast(List[Path], pipelines)
            assert all([x.is_file() for x in pipelines]), "Pipeline files must exist"
            _pipelines_paths = [make_c_string(str(s.resolve())) for s in pipelines]
            self._manager = ffi.gc(
                _api.rp_manager_create(
                    len(_pipelines_paths),
                    _pipelines_paths,
                    len(_flags),
                    _flags,
                    _workdir,
                ),
                _api.rp_manager_destroy,
            )
        else:
            pipelines = cast(List[str], pipelines)
            _pipelines = [make_c_string(s) for s in pipelines]
            self._manager = ffi.gc(
                _api.rp_manager_create_from_string(
                    len(_pipelines),
                    _pipelines,
                    len(flags),
                    flags,
                    _workdir,
                ),
                _api.rp_manager_destroy,
            )

        assert self._manager, "Failed to instantiate manager"

    def store_containers(self):
        return _api.rp_manager_store_containers(self._manager)

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

    def deserialize_target(self, serialized_target: str) -> Target:
        _target = ffi.gc(
            _api.rp_target_create_from_string(self._manager, serialized_target),
            _api.rp_target_destroy,
        )
        return Target(_target)

    def _produce_target(
        self,
        target: Target | List[Target],
        step: Step,
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
        return make_python_string(_product, True)

    def produce_target(self, target: str | List[str], step_name: str, container_name: str) -> str:
        step = self.get_step(step_name)
        if step is None:
            raise RevngException("Invalid step")

        container_identifier = self.get_container_with_name(container_name)
        if container_identifier is None:
            raise RevngException("Invalid container")

        container = step.get_container(container_identifier)
        if container is None:
            raise RevngException(f"Step {step_name} does not use container {container_name}")

        if isinstance(target, str):
            _target = self.create_target(target)
            if _target is not None:
                targets = [
                    _target,
                ]
            else:
                raise RevngException(f"Invalid target name: {_target}")
        else:
            targets = []
            for t in target:
                _t = self.create_target(t)
                if _t is not None:
                    targets.append(_t)
                else:
                    raise RevngException(f"Invalid target name: {t}")

        product = self._produce_target(targets, step, container)
        if not product:
            # TODO: we really should be able to provide a detailed error here
            raise RevngException("Failed to produce target")
        return product

    def create_target(self, target_path: str, exact: bool = True) -> Optional["Target"]:
        path, kind_name = target_path.split(":", maxsplit=1)
        path_components = path.split("/")
        kind = self.kind_from_name(kind_name)
        if kind is None:
            raise RevngException("Invalid kind")

        target = Target.create(kind, True, path_components)
        if target is None:
            raise RevngException("Invalid target")
        return target

    def recalculate_all_available_targets(self):
        _api.rp_manager_recompute_all_available_targets(self._manager)

    def get_targets_list(self, container: Container) -> Optional[TargetsList]:
        # TODO: why without this call we get back a nullptr
        # from rp_manager_get_container_targets_list?
        # It happens even if load_global_object already recomputes targets

        targets_list = _api.rp_manager_get_container_targets_list(
            self._manager, container._container
        )

        return TargetsList(targets_list) if targets_list != ffi.NULL else None

    def container_path(self, step_name: str, container_name: str) -> Optional[str]:
        _step_name = make_c_string(step_name)
        _container_name = make_c_string(container_name)
        _path = _api.rp_manager_create_container_path(self._manager, _step_name, _container_name)
        if not _path:
            return None
        path = make_python_string(_path, True)
        return path

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
        else:
            return None

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

        return [t for t in targets_list.targets()]

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
                    target_dicts = [t for t in target_list.targets()]
                    targets[step.name][container.name] = target_dicts
        return targets

    def _get_global(self, name) -> str:
        _name = make_c_string(name)
        _out = _api.rp_manager_create_serialized_global(self._manager, _name)
        return make_python_string(_out, True)

    def _set_global(self, name, content):
        _name = make_c_string(name)
        _content = make_c_string(content)
        _api.rp_manager_deserialize_global(self._manager, _content, _name)

    def get_model(self) -> str:
        return self._get_global("model.yml")

    def set_input(self, container_name: str, content: Union[bytes, str], key=None) -> str:
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
