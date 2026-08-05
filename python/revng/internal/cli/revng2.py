#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
This is just a wrapper over `pype` that sets pipebox to the revng pipebox path.
The path is computed relatively to this file, so this should work regardless of
where revng is installed.
"""

import shlex
import signal
import sys
from pathlib import Path
from typing import Any

from revng.internal.support import cache_directory
from revng.pypeline.cli.pipeline import pipeline
from revng.pypeline.cli.project import project
from revng.pypeline.cli.wrappers import WRAPPER_REGISTRY, WrapperOption
from revng.pypeline.main import main as pype_main
from revng.pypeline.main import pype
from revng.support import collect_files, get_root

from .common import ContextObject, cli_logger
from .pypeline_commands import init, quick, run_analysis_native, run_pipe_native


class ValgrindWrapperOption(WrapperOption):
    def generate_prefix(self, value: Any) -> list[str]:
        suppressions = collect_files([get_root()], ["share", "revng"], "*.supp")
        return ["valgrind", *(f"--suppressions={s}" for s in suppressions)]


class WrapperWrapperOption(WrapperOption):
    def __init__(self, name: str, help: str):  # noqa: A002
        super().__init__(name, help, type_=str)

    def generate_prefix(self, value: Any) -> list[str]:
        return shlex.split(value)


WRAPPER_REGISTRY.register_wrappers(
    WrapperOption(
        name="perf",
        help="Run program(s) under perf (for use with hotspot).",
        prefix=["perf", "record", "--call-graph", "dwarf", "--output=perf.data"],
    ),
    WrapperOption("heaptrack", help="Run program(s) under heaptrack.", prefix=["heaptrack"]),
    WrapperOption("gdb", help="Run program(s) under gdb.", prefix=["gdb", "-q", "--args"]),
    WrapperOption("lldb", help="Run program(s) under lldb.", prefix=["lldb", "--"]),
    ValgrindWrapperOption("valgrind", help="Run program(s) under valgrind."),
    WrapperOption(
        "callgrind",
        help="Run program(s) under callgrind.",
        prefix=["valgrind", "--tool=callgrind"],
    ),
    WrapperOption("rr", help="Run program(s) under rr.", prefix=["rr"]),
    WrapperWrapperOption("wrapper", help="Run program(s) with the specified wrapper."),
)


def patch_pype():
    """
    revng2 is based on `pype`, but we want to change some defaults to be revng specific,
    and we want to add some commands.
    """

    # Replace the name (needed for autocompletion and usage)
    pype.name = "revng2"

    # Patch the callback function so that the cli_logger is enabled with
    # `--verbose`
    pype_original_callback = pype.callback

    def pype_callback(*args, **kwargs):
        if kwargs["verbose"]:
            cli_logger.debug = True
        return pype_original_callback(*args, **kwargs)

    pype.callback = pype_callback

    pype.add_command(quick)
    # Replace the default for pipebox
    for param in pype.params:
        if param.name == "pipebox":
            param.default = Path(__file__).parent.parent / "pipebox.py"

    # Add `init` to project subcommand
    project.add_command(init)
    # Change the default for pipeline
    for param in project.params:
        if param.name == "pipeline":
            param.default = Path(__file__).parent.parent / "pipeline.yml"
        elif param.name == "cache_dir":
            param.default = str(cache_directory())
        elif param.name == "storage_provider":
            param.envvar = ["REVNG_STORAGE_PROVIDER", param.envvar]

    # Add native counterparts to the pipeline subcommand
    pipeline.add_command(run_pipe_native)
    pipeline.add_command(run_analysis_native)


def main():
    """Entry point for revng2."""
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(1))
    patch_pype()
    pype_main(sys.argv[1:], ContextObject)


if __name__ == "__main__":
    main()
