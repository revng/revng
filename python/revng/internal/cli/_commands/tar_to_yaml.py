#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click
import yaml

from revng.internal.cli.common import CommandRegistry
from revng.internal.cli.support import extract_tar, file_wrapper, to_string, to_yaml


def auto_process(filename: str, raw: bytes) -> str:
    # TODO: add base64 for non-text stuff
    if filename.endswith(".yml"):
        return to_yaml(filename, raw)
    else:
        return to_string(filename, raw)


@click.group(name="tar", help="Manipulate tar archives")
def tar():
    pass


@click.command(name="to-yaml", help="Turn a tar archive into YAML")
@click.argument("input_", metavar="[INPUT]", required=False)
@click.option("-o", "--output", metavar="FILE", help="Output file (stdout if omitted)")
def tar_to_yaml(input_: str | None, output: str | None) -> int:
    with file_wrapper(input_, "rb") as input_file:
        result = yaml.dump(extract_tar(input_file.read(), auto_process))
    with file_wrapper(output, "w") as output_file:
        output_file.write(result)
    return 0


def setup(registry: CommandRegistry):
    registry.register((), tar)
    registry.register(("tar",), tar_to_yaml)
