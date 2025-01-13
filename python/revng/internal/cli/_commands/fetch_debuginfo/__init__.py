#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import struct
from pathlib import Path

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import log_error

from .common import log, logger
from .elf import fetch_dwarf
from .pe import fetch_pdb

default_elf_debug_server = "https://debuginfod.elfutils.org/"
microsoft_symbol_server_url = "https://msdl.microsoft.com/download/symbols/"


def log_success(file_path):
    log(f"Result: {file_path}")


def log_fail():
    log("Result: No debug info found")


def is_pe(file):
    file.seek(0, os.SEEK_SET)
    mz_header = file.read(2)
    if mz_header != b"MZ":
        return False
    file.seek(0x3C, os.SEEK_SET)  # 0x3C contains PE offset
    pe_offset = struct.unpack("<i", file.read(4))[0]  # Offset is encoded as a little-endian int
    file.seek(pe_offset, os.SEEK_SET)
    return file.read(4) == b"PE\x00\x00"


def is_elf(file):
    file.seek(0, os.SEEK_SET)
    return file.read(4) == b"\x7fELF"


class FetchDebugInfoCommand(Command):
    def __init__(self):
        super().__init__(("model", "fetch-debuginfo"), "Fetch Debugging Information.")

    def register_arguments(self, parser):
        parser.add_argument("input", help="The input file.")
        parser.add_argument(
            "urls",
            help="List of URLs where to try finding the debug symbols.",
            nargs="*",
            default=[],
        )

    def run(self, options: Options):
        args = options.parsed_args
        logger.verbose = args.verbose

        path = Path(args.input)
        if not path.exists():
            log_error("Could not find " + args.input)
            return 1

        with path.open(mode="rb") as f:
            # Check if it is an ELF.
            if is_elf(f):
                try:
                    the_elffile = ELFFile(f)
                    if not args.urls:
                        args.urls.append(default_elf_debug_server)
                    result = fetch_dwarf(the_elffile, path, args.urls)
                except ELFError as elf_error:
                    log_error(str(elf_error))
                    return 1
            elif is_pe(f):
                # TODO: handle _NT_SYMBOL_PATH and _NT_ALT_SYMBOL_PATH
                # https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/symbol-path
                if not args.urls:
                    args.urls.append(microsoft_symbol_server_url)
                result = fetch_pdb(path, args.urls)
            else:
                log_fail()
                return 1

        if result is None:
            # We have not found the debug info file.
            log_fail()
            return 1

        log_success(result)
        return 0


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(FetchDebugInfoCommand())
