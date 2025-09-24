#
# This file is distributed under the MIT License. See LICENSE.md for details.
#


def is_mime_type_text(mime_type: str) -> bool:
    """
    Returns True if the mime type represents a text file, False otherwise.
    """
    return mime_type.startswith("text/") or mime_type in {
        "application/json",
        "application/xml",
        "application/x-yaml",
    }
