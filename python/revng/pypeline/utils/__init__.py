#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import cabc, default_dict_from_key, registry  # noqa: F401


def is_mime_type_text(mime_type: str) -> bool:
    """
    Returns True if the mime type represents a text file, False otherwise.
    """
    return mime_type.startswith("text/") or mime_type in {
        "application/json",
        "application/xml",
        "application/x-yaml",
    }
