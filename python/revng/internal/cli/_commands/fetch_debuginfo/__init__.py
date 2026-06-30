#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from argparse import RawDescriptionHelpFormatter
from pathlib import Path

from revng.internal.cli.commands_registry import Command, CommandsRegistry
from revng.internal.cli.commands_registry import Options as CLIOptions
from revng.internal.support import cache_directory, configuration
from revng.support import log_error

from .common import Options, fetch_debuginfo, log, logger


def get_config(config_key: str) -> list[str] | None:
    the_configuration = configuration()
    if (
        "debug-info-server-urls" not in the_configuration
        or config_key not in the_configuration["debug-info-server-urls"]
    ):
        return None

    result = the_configuration["debug-info-server-urls"][config_key]
    assert isinstance(result, list)
    assert all(isinstance(entry, str) for entry in result)
    return result


# This registers the `revng model fetch-debuginfo` command, while `common.py`
# file can be used as a standalone script the revng counterpart has
# revng-specific defaults (e.g. cache directory location) set.
class FetchDebugInfoCommand(Command):
    def __init__(self):
        super().__init__(("model", "fetch-debuginfo"), "Fetch Debugging Information.")
        self.options = Options(output_dir=cache_directory())
        if (config_elf_servers := get_config("dwarf")) is not None:
            self.options.elf_servers = tuple(config_elf_servers)
        if (config_pe_servers := get_config("pe")) is not None:
            self.options.pe_servers = tuple(config_pe_servers)

    def register_arguments(self, parser):
        parser.formatter_class = RawDescriptionHelpFormatter
        parser.description = """Fetches debugging symbols from the internet.
Remote urls can be overridden via revng.yml
"""
        parser.add_argument("input", help="The input file.")

    def run(self, options: CLIOptions):
        args = options.parsed_args
        logger.verbose = args.verbose

        if not Path(args.input).exists():
            log_error("Could not find " + args.input)
            return 1

        result = fetch_debuginfo(args.input, self.options)
        if result is None:
            log("Result: No debug info found")
            return 1
        else:
            log(f"Result: {result}")
            return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(FetchDebugInfoCommand())
