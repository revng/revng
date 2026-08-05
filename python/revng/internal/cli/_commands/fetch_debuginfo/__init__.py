#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path

import click

from revng.internal.cli.common import CommandRegistry, cli_logger
from revng.internal.support import cache_directory, configuration
from revng.support import log_error

from .common import Options, fetch_debuginfo


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


def _make_options() -> Options:
    options = Options(output_dir=cache_directory())
    if (config_elf_servers := get_config("dwarf")) is not None:
        options.elf_servers = tuple(config_elf_servers)
    if (config_pe_servers := get_config("pe")) is not None:
        options.pe_servers = tuple(config_pe_servers)
    return options


# This registers the `revng model fetch-debuginfo` command, while `common.py`
# file can be used as a standalone script the revng counterpart has
# revng-specific defaults (e.g. cache directory location) set.
@click.command(name="fetch-debuginfo")
@click.argument("input_", metavar="INPUT")
def fetch_debug_info(input_: str) -> int:
    """Fetches debugging symbols from the internet

    Remote urls can be overridden via revng.yml
    """
    if not Path(input_).exists():
        log_error("Could not find " + input_)
        return 1

    result = fetch_debuginfo(input_, _make_options())
    if result is None:
        cli_logger.debug_log("Result: No debug info found")
        return 1
    else:
        cli_logger.debug_log(f"Result: {result}")
        return 0


def setup(registry: CommandRegistry):
    registry.register(("model",), fetch_debug_info)
