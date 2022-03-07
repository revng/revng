#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
from typing import List, Iterable, Optional

from ._capi import _api
from .container import ContainerIdentifier, Container, TargetsList
from .kind import Kind
from .step import Step
from .target import Target
from .utils import make_c_string, make_python_string

INVALID_INDEX = 0xFFFFFFFFFFFFFFFF


class Manager:
    def __init__(
        self,
        flags: List[str],
        workdir: str,
        *,
        pipelines_paths: Optional[List[str]] = None,
        pipelines: Optional[List[str]] = None,
    ):
        if pipelines_paths is None:
            pipelines_paths = []
        _pipelines_paths = [make_c_string(s) for s in pipelines_paths]

        if pipelines is None:
            pipelines = []
        _pipelines = [make_c_string(s) for s in pipelines]

        _flags = [make_c_string(s) for s in flags]
        _workdir = make_c_string(workdir)

        # Ensures that the _manager property is always defined even if the API
        # call fails
        self._manager = None

        assert (
            bool(_pipelines) + bool(_pipelines_paths) == 1
        ), "Provide non-empty pipelines OR pipelines_paths!"

        if _pipelines_paths:
            self._manager = _api.rp_manager_create(
                len(_pipelines_paths),
                _pipelines_paths,
                len(_flags),
                _flags,
                _workdir,
            )
        else:
            self._manager = _api.rp_manager_create_from_string(
                len(_pipelines),
                _pipelines,
                len(flags),
                flags,
                _workdir,
            )

        assert self._manager, "Failed to instantiate manager"

    def store_containers(self):
        return _api.rp_manager_store_containers(self._manager)

    @property
    def kinds_count(self):
        return _api.rp_manager_kinds_count(self._manager)

    def _get_kind_from_index(self, idx: int) -> Optional[Step]:
        kind = _api.rp_manager_get_kind(self._manager, idx)

        if not kind:
            return None

        return Kind(kind)

    def kinds(self):
        for kind_idx in range(self.kinds_count):
            yield self._get_kind_from_index(kind_idx)

    def kind_from_name(self, kind_name: str) -> Optional[Kind]:
        _kind_name = make_c_string(kind_name)
        kind = _api.rp_manager_get_kind_from_name(self._manager, _kind_name)
        if not kind:
            return None
        return Kind(kind)

    def deserialize_target(self, serialized_target: str) -> Target:
        _target = _api.rp_target_create_from_string(self._manager, serialized_target)
        return Target(_target, _free_on_delete=True)

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

    def get_targets_list(self, container: Container):
        # TODO: why without this call we get back a nullptr from rp_manager_get_container_targets_list?
        #  It happens even if load_global_object already recomputes targets
        self.recalculate_all_available_targets()

        targets_list = _api.rp_manager_get_container_targets_list(
            self._manager, container._container
        )

        if not targets_list:
            return None

        return TargetsList(targets_list)

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
    def containers_count(self):
        return _api.rp_manager_containers_count(self._manager)

    def containers(self) -> Iterable[ContainerIdentifier]:
        for idx in range(self.containers_count):
            yield self._get_container_identifier(idx)

    def get_container_with_name(self, name) -> Optional[ContainerIdentifier]:
        for container in self.containers():
            if container.name == name:
                return container

    def _get_container_identifier(self, idx: int):
        container_identifier = _api.rp_manager_get_container_identifier(self._manager, idx)
        if not container_identifier:
            return None
        return ContainerIdentifier(container_identifier)

    @property
    def steps_count(self):
        return _api.rp_manager_steps_count(self._manager)

    def get_step(self, step_name: str) -> Optional[Step]:
        step_index = self._step_name_to_index(step_name)
        step = self._get_step_from_index(step_index)
        if not step:
            return None
        return step

    def steps(self):
        for step_idx in range(self.steps_count):
            yield self._get_step_from_index(step_idx)

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

    def __del__(self):
        if self._manager is not None:
            _api.rp_manager_destroy(self._manager)
