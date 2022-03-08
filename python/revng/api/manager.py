#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from pathlib import Path
from typing import List, Optional, Union, Generator, cast

from ._capi import _api, ffi
from .container import ContainerIdentifier, Container
from .kind import Kind
from .step import Step
from .target import Target, TargetsList
from .utils import make_c_string, make_python_string

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
        for kind_idx in range(self.kinds_count):
            kind = self._get_kind_from_index(kind_idx)
            if kind is not None:
                yield kind

    def kind_from_name(self, kind_name: str) -> Optional[Kind]:
        _kind_name = make_c_string(kind_name)
        kind = _api.rp_manager_get_kind_from_name(self._manager, _kind_name)
        if not kind:
            return None
        return Kind(kind)

    def deserialize_target(self, serialized_target: str) -> Target:
        _target = ffi.gc(
            _api.rp_target_create_from_string(self._manager, serialized_target),
            _api.rp_target_destroy,
        )
        return Target(_target)

    # TODO: method to produce targets in bulk (API already supports it)
    def produce_target(
        self,
        target: Target,
        step: Step,
        container: Container,
    ) -> bool:
        _targets = [target._target]
        _step = step._step
        _container = container._container
        return _api.rp_manager_produce_targets(
            self._manager, len(_targets), _targets, _step, _container
        )

    def recalculate_all_available_targets(self):
        _api.rp_manager_recompute_all_available_targets(self._manager)

    def get_targets_list(self, container: Container) -> Optional[TargetsList]:
        # TODO: why without this call we get back a nullptr
        # from rp_manager_get_container_targets_list?
        # It happens even if load_global_object already recomputes targets
        self.recalculate_all_available_targets()

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
        path = make_python_string(_path)
        _api.rp_string_destroy(_path)
        return path

    @property
    def containers_count(self) -> int:
        return _api.rp_manager_containers_count(self._manager)

    def containers(self) -> Generator[ContainerIdentifier, None, None]:
        for idx in range(self.containers_count):
            container_identifier = self._get_container_identifier(idx)
            if container_identifier is not None:
                yield container_identifier

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
        for step_idx in range(self.steps_count):
            step = self._get_step_from_index(step_idx)
            if step is not None:
                yield step

    def _step_name_to_index(self, name: str) -> Optional[int]:
        _name = make_c_string(name)
        index = _api.rp_manager_step_name_to_index(self._manager, _name)
        if index == INVALID_INDEX:
            return None
        return index

    def _get_step_from_index(self, idx: int) -> Optional[Step]:
        step = _api.rp_manager_get_step(self._manager, idx)

        if not step:
            return None

        return Step(step)
