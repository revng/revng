#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from pathlib import Path

from xdg import xdg_cache_home


def cache_directory() -> Path:
    if "PYPELINE_CACHE_DIR" in os.environ:
        return Path(os.environ["PYPELINE_CACHE_DIR"])
    else:
        return xdg_cache_home() / "pypeline"
