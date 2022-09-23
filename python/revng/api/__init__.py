#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
# flake8: noqa: F401

import atexit
from tempfile import mkdtemp
from typing import Optional

from revng.support import AnyPath

from ._capi import initialize, shutdown
from .manager import Manager
from .rank import Rank

_initialized = False


def make_manager(workdir: Optional[AnyPath] = None):
    global _initialized
    if not _initialized:
        initialize()
        atexit.register(shutdown)
        _initialized = True
    return Manager(workdir)
