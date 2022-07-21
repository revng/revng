#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shlex
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from sys import executable as py_executable

from revng.cli.commands_registry import Command, Options
from revng.cli.support import collect_libraries, handle_asan, run


class DaemonCommand(Command):
    def __init__(self):
        super().__init__(("daemon",), "Run revng GraphQL server")

    def register_arguments(self, parser: ArgumentParser):
        parser.formatter_class = RawDescriptionHelpFormatter
        parser.description = """Run revng GraphQL server

This command starts the execution of the revng graphql server with all installed
analysis libraries and pipelines.

Environment variables that are used:

STARLETTE_DEBUG: if set to "1" enables debug mode, unset when using --production
REVNG_NOTIFY_FIFOS: list of fifo pipes to be notified of changes in the pipeline
  it follows the following format:
  <fifo path 1>:<event type 1>,<fifo path 2>:<event type 2>,...
  Event types that are available: begin, context
REVNG_ORIGINS: comma-separated list of allowed CORS origins

Persistence:
revng needs a directory to preserve progress across restarts, this is controlled
by the environment variables REVNG_DATA_DIR and REVNG_PROJECT_ID
Neither of them set: use a uniquely generated temporary directory
REVNG_DATA_DIR set, REVNG_PROJECT_ID unset: use '$REVNG_DATA_DIR' as persistance folder
REVNG_PROJECT_ID set, REVNG_DATA_DIR unset: use '$XDG_DATA_HOME/revng/$REVNG_PROJECT_ID'
REVNG_DATA_DIR and REVNG_PROJECT_ID set: use '$REVNG_DATA_DIR/$REVNG_PROJECT_ID'
"""

        parser.add_argument("-p", "--port", type=str, default="8000", help="Port to use")
        parser.add_argument(
            "--hypercorn-args", type=str, default="", help="Extra arguments to pass to hypercorn"
        )
        parser.add_argument(
            "--production",
            action="store_true",
            help="Start server in production mode, "
            + "this runs the server on all interfaces and disables debug information",
        )

    def run(self, options: Options):
        _, dependencies = collect_libraries(options.search_prefixes)
        prefix = handle_asan(dependencies, options.search_prefixes)

        env = {**os.environ}
        args = ["-k", "asyncio", "-w", "1"]
        extra_args = shlex.split(options.parsed_args.hypercorn_args)

        if options.parsed_args.production:
            args.extend(["-b", f"0.0.0.0:{options.parsed_args.port}"])
        else:
            args.extend(["-b", f"127.0.0.1:{options.parsed_args.port}", "--debug"])
            env["STARLETTE_DEBUG"] = "1"

        run(
            prefix
            + [
                py_executable,
                "-m",
                "hypercorn",
                *args,
                *extra_args,
                "revng.daemon:app",
            ],
            options,
            env,
            True,
        )
