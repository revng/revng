#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from abc import ABC, abstractmethod
from io import TextIOWrapper


class Console(ABC):
    @abstractmethod
    def print(self, *args, **kwargs):  # noqa: A003
        pass


class PlainConsole(Console):
    def __init__(self, string: TextIOWrapper) -> None:
        self.string: TextIOWrapper = string

    def print(self, text: str, *, end: str = "\n", **kwargs) -> None:  # noqa: A003
        self.string.write(f"{text}{end}")

    def __del__(self):
        self.string.flush()


class StringConsole(Console):
    def __init__(self) -> None:
        self.string: str = ""

    def print(self, text: str, *, end: str = "\n", **kwargs):  # noqa: A003
        self.string += f"{text}{end}"
