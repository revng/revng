#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from .support import build_command_with_loads, run
from .commands_registry import Command, commands_registry, Options


class IROptCommand(Command):
    def __init__(self):
        super().__init__(("opt",), "LLVM's opt with rev.ng passes")

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        opt_command = build_command_with_loads(
            "opt", options.remaining_args + ["-serialize-model"], options
        )
        return run(opt_command, options)


commands_registry.register_command(IROptCommand())
