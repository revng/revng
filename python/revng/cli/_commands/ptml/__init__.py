#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.cli.commands_registry import CommandsRegistry

from .cat import PTMLCatCommand
from .strip import PTMLStripCommand


def setup(commands_registry: CommandsRegistry):
    commands_registry.define_namespace(("ptml",), "Manipulate PTML files")
    commands_registry.register_command(PTMLCatCommand())
    commands_registry.register_command(PTMLStripCommand())
