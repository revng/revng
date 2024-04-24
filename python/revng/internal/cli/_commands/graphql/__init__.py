#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import asyncio
import os
import sys
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from typing import List

from revng.internal.cli.commands_registry import Command, CommandsRegistry, Options

from .daemon_handler import DaemonHandler, ExternalDaemonHandler, InternalDaemonHandler
from .runner import Runner, produce_artifacts, run_analyses_lists, run_on_daemon, upload_file


class GraphQLCommand(Command):
    FILTER_ENV = [
        "STARLETTE_DEBUG",
        "REVNG_NOTIFY_FIFOS",
        "REVNG_ORIGINS",
        "REVNG_DATA_DIR",
        "REVNG_PROJECT_ID",
    ]

    def __init__(self):
        super().__init__(("graphql",), "Perform operations on a revng-daemon via GraphQL")

    def register_arguments(self, parser: ArgumentParser):
        parser.add_argument(
            "-e",
            "--external",
            help="Use the specified external address instead of using the local daemon",
        )
        parser.add_argument(
            "--produce-artifacts", action="store_true", help="Produce all possible artifacts"
        )
        parser.add_argument(
            "--filter-artifacts",
            action="append",
            help="Only produce artifacts with the specified component",
        )
        parser.add_argument(
            "--analyses-list", action="append", help="Analyses lists to run after upload"
        )
        parser.add_argument(
            "executable", metavar="EXECUTABLE", help="Executable to run the test with"
        )

    def run(self, options: Options):
        args = options.parsed_args

        if not args.produce_artifacts and args.filter_artifacts is not None:
            self.log("Specifying --filter-artifacts without --produce-artifacts is unsupported")
            return 1

        if args.external is not None:
            url = args.external
            self.daemon_handler: DaemonHandler = ExternalDaemonHandler(url)
        else:
            temp_folder = TemporaryDirectory()
            url = f"unix:{temp_folder.name}/daemon.sock"

            env = {k: v for k, v in os.environ.items() if k not in self.FILTER_ENV}
            self.daemon_handler = InternalDaemonHandler(url, options, env)

        executable_path = args.executable
        assert os.path.isfile(executable_path), "Executable file not found"
        runners: List[Runner] = [upload_file(executable_path)]

        if args.analyses_list is not None:
            runners.append(run_analyses_lists(args.analyses_list))

        if args.produce_artifacts:
            runners.append(produce_artifacts(args.filter_artifacts))

        run_rc = asyncio.run(run_on_daemon(self.daemon_handler, runners))
        rc = self.daemon_handler.terminate()
        if rc != 0:
            self.log(f"Daemon exited with code {rc}")

        return abs(run_rc) + abs(rc)

    @staticmethod
    def log(string: str):
        sys.stderr.write(f"{string}\n")
        sys.stderr.flush()


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(GraphQLCommand())
