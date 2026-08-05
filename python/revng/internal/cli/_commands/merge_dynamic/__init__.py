#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import click

from revng.internal.cli.common import CommandRegistry
from revng.internal.cli.support import file_wrapper

from .merge_dynamic import merge_dynamic


@click.command(
    name="merge-dynamic",
    help="Merge the dynamic portions of the translate ELF with the one from the host ELF",
)
@click.argument("to_extend", metavar="TO_EXTEND")
@click.argument("source", metavar="SOURCE")
@click.argument("output", metavar="[OUTPUT]", required=False)
@click.option(
    "--base",
    metavar="ADDRESS",
    default="0x400000",
    show_default=True,
    help="The base address where dynamic object have been loaded.",
)
@click.option(
    "--merge-load-segments",
    is_flag=True,
    help="Merge the LOADed segments from the source ELF into the output ELF.",
)
def merge_dynamic_command(
    to_extend: str,
    source: str,
    output: str | None,
    base: str,
    merge_load_segments: bool,
) -> int:
    base_address = int(base, base=0)

    with open(source, "rb") as source_file, open(to_extend, "rb") as to_extend_file, file_wrapper(
        output, "wb"
    ) as output_file:
        retcode = merge_dynamic(
            to_extend_file,
            source_file,
            output_file,
            base=base_address,
            merge_load_segments=merge_load_segments,
        )
        output_file.flush()

    return retcode


def setup(registry: CommandRegistry):
    registry.register((), merge_dynamic_command)
