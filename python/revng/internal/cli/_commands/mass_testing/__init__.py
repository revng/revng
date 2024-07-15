#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from ...commands_registry import CommandsRegistry
from .configure import MassTestingConfigureCommand
from .generate_report import MassTestingGenerateReportCommand
from .run import MassTestingRunCommand


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(MassTestingConfigureCommand())
    commands_registry.register_command(MassTestingRunCommand())
    commands_registry.register_command(MassTestingGenerateReportCommand())
