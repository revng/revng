#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
from pathlib import Path

import click

from revng.internal.cli.common import CommandRegistry
from revng.support import get_search_prefixes


@click.command(
    name="find-path",
    help=(
        "Walk revng's search prefixes (REVNG_RESOURCES env var and the "
        "running install root) and print the absolute path of the first "
        "prefix where the given relative file exists. Exits non-zero if "
        "no prefix contains it."
    ),
)
@click.argument("relpath", metavar="RELPATH")
def find_path(relpath: str) -> int:
    if Path(relpath).is_absolute():
        sys.stderr.write(f"find-path: argument must be relative: {relpath}\n")
        return 1
    prefixes = get_search_prefixes()
    for prefix in prefixes:
        candidate = Path(prefix) / relpath
        if candidate.exists():
            sys.stdout.write(str(candidate) + "\n")
            return 0
    sys.stderr.write(
        f"find-path: {relpath!r} not found under any of:\n  " + "\n  ".join(prefixes) + "\n"
    )
    return 1


def setup(registry: CommandRegistry):
    registry.register(("internal",), find_path)
