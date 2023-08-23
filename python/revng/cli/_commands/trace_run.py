#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.cli.commands_registry import Command, CommandsRegistry, Options
from revng.cli.support import get_command, handle_asan, run
from revng.support.collect import collect_libraries, collect_pipelines


class TraceRunCommand(Command):
    def __init__(self):
        super().__init__(("trace", "run"), "revng-trace-run wrapper", False)

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        cmd = get_command("revng-trace-run", options.search_prefixes)
        libraries, dependencies = collect_libraries(options.search_prefixes)
        asan_prefix = handle_asan(dependencies, options.search_prefixes)
        pipelines = collect_pipelines(options.search_prefixes)

        args_combined = [
            *[f"-load={p}" for p in libraries],
            *[f"-pipeline-path={p}" for p in pipelines],
            *options.remaining_args,
        ]

        return run([*asan_prefix, cmd, *args_combined], options)


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(TraceRunCommand())
