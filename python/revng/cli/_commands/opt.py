#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.cli.commands_registry import Command, Options
from revng.cli.support import build_command_with_loads, run


class IROptCommand(Command):
    def __init__(self):
        super().__init__(("opt",), "LLVM's opt with rev.ng passes", False)

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        opt_command = build_command_with_loads(
            "opt", options.remaining_args + ["-serialize-model"], options
        )
        return run(opt_command, options)
