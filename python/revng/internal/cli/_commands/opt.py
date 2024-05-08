#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import build_command_with_loads, run


class IROptCommand(Command):
    def __init__(self):
        super().__init__(("opt",), "LLVM's opt with rev.ng passes", False)

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        args = options.remaining_args
        if not any(arg.startswith("-enable-new-pm") for arg in args):
            args = ["-enable-new-pm=0"] + args + ["-serialize-model"]
        opt_command = build_command_with_loads("opt", args, options)
        return run(opt_command, options)


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(IROptCommand())
