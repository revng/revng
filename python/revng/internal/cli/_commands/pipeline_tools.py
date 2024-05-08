#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import build_command_with_loads, interleave, run
from revng.internal.support.collect import collect_pipelines


class PipelineToolCommand(Command):
    def __init__(self, name: str, description: str):
        super().__init__((name,), description, False)

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        pipelines = collect_pipelines(options.search_prefixes)
        pipelines_args = interleave(pipelines, "-P")
        command = build_command_with_loads(
            f"revng-{self.name}", pipelines_args + options.remaining_args, options
        )

        return run(command, options)


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(PipelineToolCommand("artifact", "revng artifact producer"))
    commands_registry.register_command(PipelineToolCommand("analyze", "revng analyzer"))
    commands_registry.register_command(PipelineToolCommand("invalidate", "revng invalidate"))
    commands_registry.register_command(PipelineToolCommand("pipeline", "revng pipeline"))
