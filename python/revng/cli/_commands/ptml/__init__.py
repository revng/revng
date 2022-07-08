#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.cli.commands_registry import commands_registry

from .cat import PTMLCatCommand  # noqa: F401
from .strip import PTMLStripCommand  # noqa: F401

commands_registry.define_namespace(("ptml",), "Manipulate PTML files")
