#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path
from typing import Dict, Generator, Optional, Union

from ._capi import _api, ffi
from .analysis import Analysis
from .container import Container, ContainerIdentifier
from .kind import Kind
from .utils import make_c_string, make_generator, make_python_string


class Step:
    def __init__(self, step):
        self._step = step

    @property
    def name(self) -> str:
        name = _api.rp_step_get_name(self._step)
        return make_python_string(name)

    @property
    def component(self) -> str:
        component = _api.rp_step_get_component(self._step)
        return make_python_string(component)

    def save(self, destination_directory: Union[Path, str]):
        dest_dir = Path(destination_directory).resolve()
        _dest_dir = make_c_string(str(dest_dir))
        return _api.rp_step_save(self._step, _dest_dir)

    def get_parent(self) -> Optional["Step"]:
        _step = _api.rp_step_get_parent(self._step)
        return Step(_step) if _step != ffi.NULL else None

    def get_container(self, container_identifier: ContainerIdentifier) -> Optional[Container]:
        _container = _api.rp_step_get_container(
            self._step, container_identifier._container_identifier
        )
        return Container(_container, self.name) if _container != ffi.NULL else None

    def get_artifacts_kind(self) -> Optional[Kind]:
        _kind = _api.rp_step_get_artifacts_kind(self._step)
        return Kind(_kind) if _kind != ffi.NULL else None

    def get_artifacts_container(self) -> Optional[Container]:
        _container = _api.rp_step_get_artifacts_container(self._step)
        return Container(_container, self.name) if _container != ffi.NULL else None

    def get_artifacts_single_target_filename(self) -> Optional[str]:
        _filename = _api.rp_step_get_artifacts_single_target_filename(self._step)
        return make_python_string(_filename) if _filename != ffi.NULL else None

    def analyses_count(self) -> int:
        return _api.rp_step_get_analyses_count(self._step)

    def _get_analysis_from_index(self, idx: int) -> Optional[Analysis]:
        _analysis = _api.rp_step_get_analysis(self._step, idx)
        return Analysis(_analysis) if _analysis != ffi.NULL else None

    def analyses(self) -> Generator[Analysis, None, None]:
        return make_generator(self.analyses_count(), self._get_analysis_from_index)

    def as_dict(self) -> Dict[str, str]:
        ret = {"name": self.name, "component": self.component}

        parent = self.get_parent()
        if parent is not None:
            ret["parent"] = parent.name

        return ret
