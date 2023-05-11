#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shlex
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from sys import executable as py_executable

from revng.cli.commands_registry import Command, CommandsRegistry, Options
from revng.cli.support import collect_libraries, exec_run, handle_asan


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
STARLETTE_MIDDLEWARES_{EARLY,LATE}: a comma-separated list of extra Middlewares to use
in starlette. It follows the syntax:
  import.path.to.module.ObjectName,...
REVNG_NOTIFY_FIFOS: list of fifo pipes to be notified of changes in the pipeline
  it follows the following format:
  <fifo path 1>:<event type 1>,<fifo path 2>:<event type 2>,...
  Event types that are available: begin, context
REVNG_ORIGINS: comma-separated list of allowed CORS origins
REVNG_C_API_TRACE_PATH: path to file to use to save api tracing, useful for debugging

Persistence:
revng needs a directory to preserve progress across restarts, this is controlled
by the environment variables REVNG_DATA_DIR and REVNG_PROJECT_ID
Neither of them set: use a uniquely generated temporary directory
REVNG_DATA_DIR set, REVNG_PROJECT_ID unset: use '$REVNG_DATA_DIR' as persistence folder
REVNG_PROJECT_ID set, REVNG_DATA_DIR unset: use '$XDG_DATA_HOME/revng/$REVNG_PROJECT_ID'
REVNG_DATA_DIR and REVNG_PROJECT_ID set: use '$REVNG_DATA_DIR/$REVNG_PROJECT_ID'
"""

        parser.add_argument("-p", "--port", type=int, default=8000, help="Port to use")
        parser.add_argument(
            "--uvicorn-args", type=str, default="", help="Extra arguments to pass to uvicorn"
        )
        parser.add_argument(
            "--production",
            action="store_true",
            help="Start server in production mode, this runs the server on all interfaces",
        )
        parser.add_argument(
            "-b",
            "--bind",
            type=str,
            help=(
                "Manually bind to the specified address. "
                "format: 'tcp:<ip>:port'/'unix:<path>' or 'none' to disable binding"
            ),
        )

    def run(self, options: Options):
        _, dependencies = collect_libraries(options.search_prefixes)
        prefix = handle_asan(dependencies, options.search_prefixes)

        env = {**os.environ}
        args = ["--no-access-log", "--workers", "1", "--timeout-keep-alive", "600"]
        extra_args = shlex.split(options.parsed_args.uvicorn_args)

        host = "127.0.0.1"
        port = str(options.parsed_args.port)
        if bind := options.parsed_args.bind:
            if bind.startswith("tcp:"):
                _, host, port = bind.split(":", 2)
                args.extend(["--host", host, "--port", port])
            elif bind.startswith("unix:"):
                _, path = bind.split(":", 1)
                args.extend(["--uds", path])
            elif bind == "none":
                pass
            else:
                raise ValueError(f"Unknown bind address: {bind}")
        else:
            if options.parsed_args.production:
                host = "0.0.0.0"
            args.extend(["--host", host, "--port", port])

        if not options.parsed_args.production:
            env["STARLETTE_DEBUG"] = "1"
            env["REVNG_ORIGINS"] = "*"

        exec_run(
            prefix
            + [
                py_executable,
                "-m",
                "uvicorn",
                *args,
                *extra_args,
                "revng.daemon:app",
            ],
            options,
            env,
        )


def setup(commands_registry: CommandsRegistry):
    commands_registry.register_command(DaemonCommand())
