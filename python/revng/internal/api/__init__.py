#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import atexit
from typing import Optional

from revng.support import AnyPath

from ._capi import initialize, shutdown
from .manager import Manager

__all__ = ["Manager", "initialize", "shutdown"]

_initialized = False


def make_manager(workdir: Optional[AnyPath] = None):
    global _initialized
    if not _initialized:
        initialize()
        atexit.register(shutdown)
        _initialized = True

    if workdir is None:
        return Manager(None)
    else:
        return Manager(str(workdir))
