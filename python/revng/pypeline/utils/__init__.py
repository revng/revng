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


def is_mime_type_text(mime_type: str) -> bool:
    """
    Returns True if the mime type represents a text file, False otherwise.
    """
    return mime_type.startswith("text/") or mime_type in {
        "application/json",
        "application/xml",
        "application/x-yaml",
    }
