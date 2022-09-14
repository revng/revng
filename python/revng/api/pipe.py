#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import Any, Dict, Generator, Optional

from ._capi import _api, ffi
from .kind import Kind
from .utils import make_python_string


class Pipe:
    def __init__(self, pipe):
        self._pipe = pipe

    @property
    def name(self) -> str:
        _name = _api.rp_pipe_get_name(self._pipe)
        return make_python_string(_name)

    @property
    def doc(self) -> str:
        _doc = _api.rp_pipe_get_doc(self._pipe)
        return make_python_string(_doc)


    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "doc": self.doc
        }

    def __hash__(self):
        return id(self)
