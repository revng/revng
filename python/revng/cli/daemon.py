#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from sys import executable as py_executable

from .commands_registry import Command, Options, commands_registry
from .support import collect_files, collect_libraries, run


class DaemonCommand(Command):
    def __init__(self):
        super().__init__(("daemon",), "Run revng GraphQL server")

    def register_arguments(self, parser):
        parser.add_argument("-p", "--port", type=str)

    def run(self, options: Options):
        libraries, _ = collect_libraries(options.search_prefixes)
        pipelines = collect_files(options.search_prefixes, ["share", "revng", "pipelines"], "*.yml")

        if options.parsed_args.port is not None:
            port = options.parsed_args.port
        else:
            port = "8000"

        env = {
            **os.environ,
            "REVNG_ANALYSIS_LIBRARIES": ":".join(libraries),
            "REVNG_PIPELINES": ",".join(pipelines),
            "FLASK_RUN_PORT": port,
            "FLASK_APP": "revng.daemon",
            "FLASK_ENV": "development",
        }
        return run([py_executable, "-m", "flask", "run", "--no-reload"], options, env)


commands_registry.register_command(DaemonCommand())
