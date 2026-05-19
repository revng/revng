#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import tarfile
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from base64 import b64decode
from contextlib import suppress
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory, TemporaryFile
from typing import IO

import yaml

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import build_command_with_loads, file_wrapper, try_run
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


class MergeLLVMIRCommand(Command):
    def __init__(self):
        super().__init__(("merge", "llvm"), "Merge multiple LLVM IR files into a single one")

    def register_arguments(self, parser: ArgumentParser):
        parser.formatter_class = RawDescriptionHelpFormatter
        parser.description = """Links multiple LLVM IR files into a single one.
This program works differently depending on the number of inputs:
* If one input is passed it is interpreted as either:
  * a YAML dict with base64-encoded values
  * a tar where each file is a bitcode file
  * a single LLVM module
* If no input is passed, treat stdin as a single input
* If multiple input files are passed, treat all of them as bitcode files
"""
        parser.add_argument("-o", "--output", help="Where to output the file, stdout if omitted")
        parser.add_argument("inputs", nargs="*", help="Input file(s), omit to use stdin")

    def run(self, options: Options):
        args = options.parsed_args
        if len(args.inputs) >= 2:
            return self.run_merge(args.inputs, options)

        input_file = args.inputs[0] if len(args.inputs) == 1 else "-"
        with file_wrapper(input_file, "rb") as input_:
            data = read_single_file(input_)

        with TemporaryDirectory(prefix="tmp.revng-merge-llvm.") as temp_dir:
            temp_dir_path = Path(temp_dir)
            input_paths = []
            for index, value in enumerate(data):
                element_path = temp_dir_path / str(index)
                element_path.write_bytes(value)
                input_paths.append(str(element_path.resolve()))

            return self.run_merge(input_paths, options)

    def run_merge(self, inputs: list[str], options: Options):
        with file_wrapper(options.parsed_args.output, "wb") as output_file:
            return try_run(
                build_command_with_loads("merge-llvm", inputs, options),
                options,
                stdout=output_file,
            )


def setup(commands_registry: CommandsRegistry):
    commands_registry.define_namespace(("merge",), "Tools to merge function-wise object together")
    commands_registry.register_command(MergeLLVMIRCommand())
