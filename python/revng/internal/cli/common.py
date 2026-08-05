#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
Definitions shared by all the commands of the revng command-line.
"""

from typing import Callable, Concatenate, ParamSpec, Protocol, TypeVar

import click

from revng.pypeline.cli.context import ContextObject as PypelineContextObject
from revng.pypeline.cli.wrappers import WrappableCommand as PypelineWrappableCommand
from revng.pypeline.utils.logger import get_logger


class WrappableCommand(PypelineWrappableCommand):
    pass


class ContextObject(PypelineContextObject):
    pass


class _ObjectProtocol(Protocol):
    obj: ContextObject


class ClickContext(_ObjectProtocol, click.Context):
    pass


P = ParamSpec("P")
R = TypeVar("R")


def pass_context(f: Callable[Concatenate[ClickContext, P], R]) -> Callable[P, R]:
    return click.pass_context(f)  # type: ignore


cli_logger = get_logger("cli")
"""
The pre-initialized logger to use inside the revng command-line code.
"""
