#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from pathlib import Path

from revng.internal.support import import_pipebox

_native_libraries = [Path(__file__).parent / "libmockPipebox.so"]
_module, _handles = import_pipebox(_native_libraries)
_native_pipes = {
    "AppendFooPipe": _module.AppendFooPipe,
    "CreateEntryPipe": _module.CreateEntryPipe,
    "AppendFooPipe2": _module.AppendFooPipe2,
}
_native_analyses = {
    "AppendFooLibAnalysis": _module.AppendFooLibAnalysis,
    "AppendFooLibAnalysis2": _module.AppendFooLibAnalysis2,
}


def initialize(argv: list[str] = []):
    pass


def argv_hook(input_: list[str]):
    assert input_[2] in ("run-pipe", "run-analysis")
    if input_[2] == "run-pipe":
        return ["revng", "pipeline", "run-pipe-native", *input_[3:]]
    else:
        return ["revng", "pipeline", "run-analysis-native", *input_[3:]]
