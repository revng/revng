#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# flake8: noqa: F401

from tempfile import mkdtemp
from typing import Optional

from revng.support import AnyPath

from ._capi import initialize
from .manager import Manager
from .rank import Rank

_initialized = False


def make_manager(workdir: Optional[AnyPath] = None):
    global _initialized
    if not _initialized:
        initialize()
        _initialized = True
    return Manager(workdir if workdir is not None else mkdtemp())
