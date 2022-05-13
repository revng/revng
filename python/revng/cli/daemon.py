#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shlex
from argparse import ArgumentParser
from sys import executable as py_executable

from .commands_registry import Command, Options, commands_registry
from .support import collect_files, collect_libraries, handle_asan, run


class DaemonCommand(Command):
    def __init__(self):
        super().__init__(("daemon",), "Run revng GraphQL server")

    def register_arguments(self, parser: ArgumentParser):
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
        (libraries, dependencies) = collect_libraries(options.search_prefixes)
        pipelines = collect_files(options.search_prefixes, ["share", "revng", "pipelines"], "*.yml")
        prefix = handle_asan(dependencies, options.search_prefixes)

        env = {
            "REVNG_ANALYSIS_LIBRARIES": ":".join(libraries),
            "REVNG_PIPELINES": ",".join(pipelines),
        }

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
            {**os.environ, **env},
            True,
        )


commands_registry.register_command(DaemonCommand())
