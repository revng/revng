#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class FileRequest:
    # Hash of the file, this is a hex-encoded SHA256, with all letters lowercase
    hash: str  # noqa: A003
    # Name of the file, this is optional but allows fuzzy searching in some cases
    name: str | None = field(default=None)
    # Size (in bytes) of the file, this is optional but allows fuzzy searching
    # in some cases
    size: int | None = field(default=None)


class FileProvider(ABC):
    @abstractmethod
    def get_files(self, requests: list[FileRequest]) -> dict[str, bytes]:
        """Request files, return a dict <hash> -> <contents>"""
