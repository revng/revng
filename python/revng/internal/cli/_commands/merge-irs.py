#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import tarfile
from base64 import b64decode
from contextlib import suppress
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory, TemporaryFile
from typing import IO

import click
import yaml

from revng.internal.cli.common import ClickContext, CommandRegistry, WrappableCommand, pass_context
from revng.internal.cli.support import build_command_with_loads, file_wrapper
from revng.support import TarDictionary


def read_single_file(input_: IO[bytes]) -> list[bytes]:
    # If we encounter a non-seekable file (e.g. stdin) first copy it to a
    # temporary file and then re-run this function
    if not input_.seekable():
        with TemporaryFile() as temp_file:
            copyfileobj(input_, temp_file)
            temp_file.seek(0)
            return read_single_file(temp_file)

    # If here, the file is seekable, test the various formats by trying to read
    # them and moving onto the next format if the previous throws an error

    # Test for YAML
    with suppress(yaml.YAMLError):
        data = yaml.load(input_, Loader=yaml.CSafeLoader)
        return [b64decode(v) for v in data.values()]

    # Test for tar
    input_.seek(0)
    with suppress(tarfile.ReadError):
        return list(TarDictionary(input_).values())

    # No good format found, assume it's a single plain file
    input_.seek(0)
    return [input_.read()]


@click.group(name="merge", help="Tools to merge function-wise object together")
def merge():
    pass


@click.command(
    cls=WrappableCommand,
    name="llvm",
    help="""Links multiple LLVM IR files into a single one.

\b
This program works differently depending on the number of inputs:
* If one input is passed it is interpreted as either:
  * a YAML dict with base64-encoded values
  * a tar where each file is a bitcode file
  * a single LLVM module
* If no input is passed, treat stdin as a single input
* If multiple input files are passed, treat all of them as bitcode files
""",
)
@click.option("-o", "--output", help="Where to output the file, stdout if omitted")
@click.argument("inputs", metavar="[INPUTS]...", nargs=-1)
@pass_context
def merge_llvm(ctx: ClickContext, output: str | None, inputs: tuple[str, ...]) -> int:
    if len(inputs) >= 2:
        return run_merge(ctx, list(inputs), output)

    input_file = inputs[0] if len(inputs) == 1 else "-"
    with file_wrapper(input_file, "rb") as input_:
        data = read_single_file(input_)

    with TemporaryDirectory(prefix="tmp.revng-merge-llvm.") as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_paths = []
        for index, value in enumerate(data):
            element_path = temp_dir_path / str(index)
            element_path.write_bytes(value)
            input_paths.append(str(element_path.resolve()))

        return run_merge(ctx, input_paths, output)


def run_merge(ctx: ClickContext, inputs: list[str], output: str | None) -> int:
    with file_wrapper(output, "wb") as output_file:
        return ctx.obj.try_run(
            build_command_with_loads("merge-llvm", inputs),
            stdout=output_file,
        )


def setup(registry: CommandRegistry):
    registry.register((), merge)
    registry.register(("merge",), merge_llvm)
