#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import os
import sys
from argparse import ArgumentParser
from tempfile import TemporaryDirectory

from revng.cli.commands_registry import Command, CommandsRegistry, Options

from .daemon_handler import DaemonHandler, ExternalDaemonHandler, InternalDaemonHandler
from .test import run_self_test


class DaemonSelfTestCommand(Command):
    FILTER_ENV = [
        "STARLETTE_DEBUG",
        "REVNG_NOTIFY_FIFOS",
        "REVNG_ORIGINS",
        "REVNG_DATA_DIR",
        "REVNG_PROJECT_ID",
    ]

    def __init__(self):
        super().__init__(
            ("daemon-self-test",),
            "Check if revng-daemon can produce all artifacts given an executable",
        )

    def register_arguments(self, parser: ArgumentParser):
        parser.add_argument(
            "-e",
            "--external",
            help="Use the specified external address instead of using the local daemon",
        )
        parser.add_argument(
            "--revng-c",
            action="store_true",
            help="Enable additional checks, which assume revng-c is installed",
        )
        parser.add_argument(
            "executable", metavar="EXECUTABLE", help="Executable to run the test with"
        )

    def run(self, options: Options):
        args = options.parsed_args

        if args.external is not None:
            url = args.external
            self.daemon_handler: DaemonHandler = ExternalDaemonHandler(url)
        else:
            temp_folder = TemporaryDirectory()
            url = f"unix:{temp_folder.name}/daemon.sock"

            env = {k: v for k, v in os.environ.items() if k not in self.FILTER_ENV}
            self.daemon_handler = InternalDaemonHandler(url, options, env)

        self.log(f"Starting daemon self-test, url: {url}")
        executable_path = args.executable
        assert os.path.isfile(executable_path), "Executable file not found"
        asyncio.run(run_self_test(self.daemon_handler, executable_path, args.revng_c))

        rc = self.daemon_handler.terminate()
        if rc != 0:
            raise RuntimeError(f"Daemon exited with code {rc}")

    @staticmethod
    def log(string: str):
        sys.stderr.write(f"{string}\n")
        sys.stderr.flush()


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(DaemonSelfTestCommand())
