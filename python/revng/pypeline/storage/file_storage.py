#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class FileRequest:
    hash: str  # noqa: A003
    name: str | None = field(default=None)
    size: int = field(default=-1)


class FileStorage(ABC):
    @abstractmethod
    def get_files(self, requests: list[FileRequest]) -> dict[str, bytes]:
        """Request files, return a dict <hash> -> <contents>"""
