#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import signal
import sys
from subprocess import PIPE, Popen
from typing import BinaryIO, cast

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options
from revng.internal.cli.support import build_command_with_loads, popen, run


class IROptCommand(Command):
    def __init__(self):
        super().__init__(("opt",), "LLVM's opt with rev.ng passes", False)

    def register_arguments(self, parser):
        pass

    def run(self, options: Options):
        if options.dry_run:
            return 0

        args = options.remaining_args
        if not any(arg.startswith("-enable-new-pm") for arg in args):
            args = ["-enable-new-pm=0"] + args

        input_path = None
        for index, arg in enumerate(args[:]):
            if not arg.startswith("-"):
                input_path = arg
                args[index] = "-"
                break

        opt_command = build_command_with_loads("opt", args, options)
        if input_path is None and sys.stdin.isatty():
            return run(opt_command, options)

        if input_path is not None:
            input_file: BinaryIO = open(input_path, "rb")  # noqa: SIM115
        else:
            input_file = sys.stdin.buffer

        magic = input_file.read(4)
        if magic == b"\x28\xb5\x2f\xfd":
            # Zstd stream
            zstd_proc = cast(Popen, popen(["zstdcat"], options, stdin=PIPE, stdout=PIPE))
            write_target = cast(BinaryIO, zstd_proc.stdin)
            main_proc = cast(Popen, popen(opt_command, options, stdin=zstd_proc.stdout))
        else:
            zstd_proc = None
            main_proc = cast(Popen, popen(opt_command, options, stdin=PIPE))
            write_target = cast(BinaryIO, main_proc.stdin)

        write_target.write(magic)
        while True:
            data = input_file.read(4096)
            if data == b"":
                break
            write_target.write(data)
        input_file.close()
        write_target.close()

        main_proc.wait()
        if zstd_proc is not None:
            try:
                zstd_proc.wait(2)
            except TimeoutError:
                zstd_proc.send_signal(signal.SIGKILL)

        return main_proc.returncode


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(IROptCommand())
