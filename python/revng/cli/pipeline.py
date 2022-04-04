#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from .support import build_command_with_loads, run, collect_files, interleave
from .commands_registry import Command, commands_registry, Options


class PipelineCommand(Command):
    def __init__(self):
        super().__init__(("pipeline",), "revng meta pipeline", False)

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        pipelines = collect_files(options.search_prefixes, ["share", "revng", "pipelines"], "*.yml")
        pipelines_args = interleave(pipelines, "-P")
        command = build_command_with_loads(
            "revng-pipeline", pipelines_args + options.remaining_args, options
        )
        return run(command, options)


commands_registry.register_command(PipelineCommand())
