#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys

import click
import yaml

from revng.internal.cli.common import CommandRegistry

from .remap import remap_metaaddress


@click.command(name="to-json", help="Extract and process rev.ng model")
@click.option("--remap", is_flag=True, help="Remap MetaAddresses")
def model_to_json(remap: bool) -> int:
    # Consume YAML generated from revng-efa-extractcfg
    input_file = sys.stdin

    # Decode YAML
    parsed_text = yaml.load(input_file, Loader=yaml.SafeLoader)

    # Remap MetaAddress
    if remap:
        parsed_text = remap_metaaddress(parsed_text)

    # Dump as JSON
    print(yaml.dump(parsed_text))
    return 0


def setup(registry: CommandRegistry):
    registry.register(("model",), model_to_json)
