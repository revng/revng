#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import os
from functools import cached_property
from pathlib import Path

import yaml

from .stacktrace import Stacktrace, stacktrace_transform

REQUIRED_FILES = ("input", "output.log", "test-harness.json")
COPY_EXCLUDED_FILES = ("meta.yml",)


class TestDirectory:
    """Class that abstracts away aspects of a directory that has been run under
    `test-harness`. The constructor allows to bind any directory, before using
    any of its attributes, is_valid should be checked."""

    def __init__(self, raw_path: str | Path, name: str):
        self.path = Path(raw_path).resolve()
        self.name = name

    def is_valid(self):
        return all((self.path / f).is_file() for f in REQUIRED_FILES)

    @cached_property
    def test_harness_data(self) -> dict:
        with open(self.path / "test-harness.json", "r") as f:
            return json.load(f)

    @property
    def exit_code(self) -> int:
        return self.test_harness_data["exit_code"]

    @property
    def has_timed_out(self) -> bool:
        return self.test_harness_data["timeout"]

    @property
    def status(self):
        if self.has_timed_out:
            return "TIMED_OUT"
        if self.exit_code == 0:
            return "OK"
        if self.exit_code == 235:
            return "OOM"
        if not (self.path / "stacktrace.json").is_file():
            return "FAILED"
        else:
            return "CRASHED"

    @property
    def elapsed_time(self):
        return self.test_harness_data["time"]["elapsed_time"]

    @cached_property
    def stacktrace(self) -> Stacktrace | None:
        stacktrace = self.path / "stacktrace.json"
        if stacktrace.exists():
            with open(stacktrace) as f:
                return stacktrace_transform(f)
        else:
            return None

    def stacktrace_id(self) -> str:
        if self.stacktrace is not None:
            return self.stacktrace.id_
        else:
            return ""

    def has_input(self) -> bool:
        return (self.path / "input").exists()

    @cached_property
    def input_name(self) -> str:
        if self.has_input():
            return (self.path / "input").resolve().name
        else:
            return ""

    def has_trace(self) -> bool:
        return (self.path / "trace.json.gz").exists()

    @cached_property
    def meta(self) -> dict | None:
        meta_file = self.path / "meta.yml"
        if not meta_file.exists():
            return None

        with open(meta_file) as f:
            return yaml.safe_load(f)

    def get_meta(self, key: str) -> str:
        assert self.meta is not None
        return self.meta[key]

    def copy_to(self, destination: str | Path):
        destination = Path(destination)
        with os.scandir(self.path) as scanner:
            for entry in scanner:
                if not entry.is_file() or entry.name in COPY_EXCLUDED_FILES:
                    continue
                os.symlink(os.path.relpath(entry.path, destination), destination / entry.name)
