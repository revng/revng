#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import functools
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, copytree, which
from subprocess import DEVNULL, PIPE, run
from typing import Dict

import yaml

from revng.internal.cli.commands_registry import Command, Options
from revng.support import get_root

from .db import create_and_populate
from .meta import GlobalMeta
from .stacktrace import generate_crash_components, generate_flamegraph, get_filter
from .test_directory import TestDirectory


@functools.cache
def generate_inkscape_data():
    # Filter out DISPLAY, if set wrong it causes inkscape to fail
    inkscape_env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}

    # Inkscape might be >1.0, which requires -e, whereas inkscape >=1.0 requires -o
    proc = run(["inkscape", "--version"], env=inkscape_env, stdout=PIPE, text=True)
    old_inkscape = proc.stdout.startswith("Inkscape 0")
    export_switch = "-e" if old_inkscape else "-o"

    return {"export_switch": export_switch, "env": inkscape_env}


def svg_to_png(svg_input: str, png_output: str):
    inkscape_data = generate_inkscape_data()
    run(
        ("inkscape", "-w", "1920", svg_input, inkscape_data["export_switch"], png_output),
        env=inkscape_data["env"],
        stdout=DEVNULL,
        stderr=DEVNULL,
        check=True,
    )


@dataclass(frozen=True)
class GFOptions:
    file_prefix: str
    legend_prefix: str
    end_location_name: str


def generate_flamegraphs(
    stacktraces, output: Path, options: GFOptions, exclude_paths: list[re.Pattern]
):
    generate_flamegraph(
        stacktraces,
        output / f"{options.file_prefix}_topdown.svg",
        f"{options.legend_prefix} - Top down - Thread entry point is at the bottom",
        exclude_paths,
        True,
    )
    generate_flamegraph(
        stacktraces,
        output / f"{options.file_prefix}_bottomup.svg",
        f"{options.legend_prefix} - Bottom up - {options.end_location_name} location is at the top",
        exclude_paths,
    )

    for name in ("topdown", "bottomup"):
        svg_to_png(
            str(output / f"{options.file_prefix}_{name}.svg"),
            str(output / f"{options.file_prefix}_{name}.png"),
        )


class MassTestingGenerateReportCommand(Command):
    def __init__(self):
        super().__init__(("mass-testing", "generate-report"), "Generate report for mass-testing")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.description = "Generate report for mass-testing"
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")

    def run(self, options: Options):
        if which("inkscape") is None:
            sys.stderr.write("ERROR: 'inkscape' not found in PATH")
            return 1

        args = options.parsed_args
        output = Path(args.output)

        tests = []
        for dirpath, dirnames, filenames in os.walk(args.input):
            test_dir = TestDirectory(dirpath, os.path.relpath(dirpath, args.input))
            if test_dir.is_valid():
                tests.append(test_dir)

        global_meta_path = Path(args.input) / "meta.yml"
        if global_meta_path.exists():
            with open(global_meta_path) as f:
                global_meta = GlobalMeta.from_dict(yaml.safe_load(f))
        else:
            global_meta = GlobalMeta.from_dict({})

        stacktrace_categories = ("CRASHED", "TIMED_OUT", "OOM")
        tests_by_category: Dict[str, list[TestDirectory]] = {k: [] for k in stacktrace_categories}
        for test in tests:
            if test.status in stacktrace_categories:
                tests_by_category[test.status].append(test)

        stack_aggregation = global_meta.stacktrace_aggregation
        gf_options = {
            "CRASHED": GFOptions("failures", "Stack traces", "Crash"),
            "TIMED_OUT": GFOptions("timeouts", "Timeouts", "Time out"),
            "OOM": GFOptions("ooms", "OOMs", "OOM"),
        }

        total_counts: list[tuple[str, str, str, int]] = []
        flamegraph_exclude_paths = global_meta.flamegraph_exclude_paths
        for component_filter in global_meta.crash_components_filters:
            for value in component_filter.values:
                filter_ = get_filter(component_filter.type, component_filter.variable, value)
                stacktraces = filter_.filter_(tests_by_category[component_filter.category])
                suffix = filter_.suffix()
                total_counts.extend(
                    [
                        (component_filter.category, suffix, *v)
                        for v in generate_crash_components(stacktraces, stack_aggregation)
                    ]
                )

                parent_options = gf_options[component_filter.category]
                new_options = GFOptions(
                    f"{parent_options.file_prefix}_{suffix}",
                    parent_options.legend_prefix,
                    parent_options.end_location_name,
                )
                generate_flamegraphs(stacktraces, output, new_options, flamegraph_exclude_paths)

        for cat, cat_tests in tests_by_category.items():
            stacktraces = [t.stacktrace for t in cat_tests]
            total_counts.extend(
                [
                    (cat, "all", *v)
                    for v in generate_crash_components(stacktraces, stack_aggregation)
                ]
            )
            generate_flamegraphs(stacktraces, output, gf_options[cat], flamegraph_exclude_paths)

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

        return 0
