#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Location:
    """A location is a '/'-delimited string with a type at the beginning and
    one or more path components, e.g. `/function/0x1000:Code_x86_64`."""

    type: str  # noqa: A003
    path_components: List[str]

    @staticmethod
    def parse(string: str) -> Optional["Location"]:
        if not string.startswith("/"):
            return None

        parts = string.split("/")[1:]
        return Location(parts[0], parts[1:])

    def to_string(self) -> str:
        return f"/{self.type}/{'/'.join(self.path_components)}"
