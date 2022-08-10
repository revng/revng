#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.cli.commands_registry import Command, CommandsRegistry, Options
from revng.cli.support import build_command_with_loads, collect_files, interleave, run


class PipelineToolCommand(Command):
    def __init__(self, name: str, description: str):
        super().__init__((name,), description, False)

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        pipelines = collect_files(options.search_prefixes, ["share", "revng", "pipelines"], "*.yml")
        pipelines_args = interleave(pipelines, "-P")
        command = build_command_with_loads(
            f"revng-{self.name}", pipelines_args + options.remaining_args, options
        )
        return run(command, options)


class ArtifactTool(PipelineToolCommand):
    def __init__(self):
        super().__init__("artifact", "revng artifact producer")


class AnalyzeTool(PipelineToolCommand):
    def __init__(self):
        super().__init__("analyze", "revng analyzer")


class InvalidateTool(PipelineToolCommand):
    def __init__(self):
        super().__init__("invalidate", "revng invalidate")


class PipelineTool(PipelineToolCommand):
    def __init__(self):
        super().__init__("pipeline", "revng pipeline")


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(ArtifactTool())
    commands_registry.register_command(AnalyzeTool())
    commands_registry.register_command(InvalidateTool())
    commands_registry.register_command(PipelineTool())
