#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import os
import sys
from pathlib import Path
from shutil import copy2, copytree, which
from subprocess import DEVNULL, PIPE, run

import yaml

from revng.internal.cli.support import get_root

from ....commands_registry import Command, Options
from .db import create_and_populate
from .meta import GlobalMeta
from .stacktrace import generate_crash_components, generate_flamegraph
from .test_directory import TestDirectory


def generate_flamegraphs(crashed_stacktraces, timed_out_stacktraces, output, options):
    if which("inkscape") is None:
        sys.stderr.write("ERROR: 'inkscape' not found in PATH")
        sys.exit(1)

    generate_flamegraph(
        crashed_stacktraces,
        output / "failures_topdown.svg",
        "Stack traces - Top down - Thread entry point is at the bottom",
        True,
    )
    generate_flamegraph(
        crashed_stacktraces,
        output / "failures_bottomup.svg",
        "Stack traces - Bottom up - Crash location is at the top",
    )
    generate_flamegraph(
        timed_out_stacktraces,
        output / "timeouts_topdown.svg",
        "Timeouts - Top down - Thread entry point is at the bottom",
    )
    generate_flamegraph(
        timed_out_stacktraces,
        output / "timeouts_bottomup.svg",
        "Timeouts - Bottom up - Time out location is at the top",
        True,
    )

    # Filter out DISPLAY, if set wrong it causes inkscape to fail
    inkscape_env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    # Inkscape might be >1.0, which requires -e, whereas inkscape >=1.0 requires -o
    proc = run(["inkscape", "--version"], env=inkscape_env, stdout=PIPE, text=True)
    old_inkscape = proc.stdout.startswith("Inkscape 0")
    export_switch = "-e" if old_inkscape else "-o"

    svgs = ["failures_topdown", "failures_bottomup", "timeouts_topdown", "timeouts_bottomup"]
    for svg in svgs:
        run(
            ("inkscape", "-w", "1920")
            + (str(output / f"{svg}.svg"), export_switch, str(output / f"{svg}.png")),
            env=inkscape_env,
            stdout=DEVNULL,
            stderr=DEVNULL,
            check=True,
        )


class MassTestingGenerateReportCommand(Command):
    def __init__(self):
        super().__init__(("mass-testing", "generate-report"), "Generate report for mass-testing")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.description = "Generate report for mass-testing"
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")

    def run(self, options: Options):
        args = options.parsed_args
        tests = []
        for dirpath, dirnames, filenames in os.walk(args.input):
            test_dir = TestDirectory(dirpath, os.path.relpath(dirpath, args.input))
            if test_dir.is_valid():
                tests.append(test_dir)

        crashed_stacktraces = [d.stacktrace for d in tests if d.status == "CRASHED"]
        timed_out_stacktraces = [d.stacktrace for d in tests if d.status == "TIMED_OUT"]

        global_meta_path = Path(args.input) / "meta.yml"
        if global_meta_path.exists():
            with open(global_meta_path) as f:
                global_meta = GlobalMeta.from_dict(yaml.safe_load(f))
        else:
            global_meta = GlobalMeta.from_dict({})

        stack_aggregation = global_meta.stacktrace_aggregation
        crash_counts = generate_crash_components(crashed_stacktraces, stack_aggregation)
        timed_out_counts = generate_crash_components(timed_out_stacktraces, stack_aggregation)
        total_counts = {"CRASHED": crash_counts, "TIMED_OUT": timed_out_counts}

        output = Path(args.output)
        generate_flamegraphs(crashed_stacktraces, timed_out_stacktraces, output, options)

        db = output / "main.db"
        db.unlink(missing_ok=True)
        create_and_populate(db, tests, total_counts, global_meta)

        copytree(get_root() / "share/mass-testing-report", output, dirs_exist_ok=True)
        if global_meta_path.exists():
            copy2(global_meta_path, output)

        for test in tests:
            output_dir = output / test.name
            output_dir.mkdir(parents=True, exist_ok=True)
            test.copy_to(output_dir)
