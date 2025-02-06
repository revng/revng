#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import os
import time
from pathlib import Path
from subprocess import run

import yaml

from revng.internal.cli.commands_registry import Command, Options
from revng.internal.cli.support import get_root


class MassTestingRunCommand(Command):
    def __init__(self):
        super().__init__(("mass-testing", "run"), "Run a mass-testing configuration")

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.description = "Run a mass-testing configuration"
        parser.add_argument("-j", "--jobs", type=int, help="Number of parallel jobs to use")
        parser.add_argument("build_dir", help="Build directory")

    def run(self, options: Options):
        args = options.parsed_args

        if "MASS_TESTING_RUN_JOBS" in os.environ:
            cpu_count = int(os.environ["MASS_TESTING_RUN_JOBS"])
        elif args.jobs:
            cpu_count = args.jobs
        else:
            cpu_count = len(os.sched_getaffinity(0))

        meta_file = Path(args.build_dir) / "meta.yml"
        with open(meta_file) as f:
            meta_data = yaml.safe_load(f)
        if "cpu_count" not in meta_data:
            meta_data["cpu_count"] = cpu_count
        if "start_time" not in meta_data:
            meta_data["start_time"] = time.time()
        with open(meta_file, "w") as f:
            yaml.safe_dump(meta_data, f)

        new_env = os.environ.copy()
        bin_dir = get_root() / "libexec/revng/mass-testing"
        new_env["PATH"] = f"{bin_dir.resolve()!s}:{new_env['PATH']}"

        run(
            ["ninja", "--quiet", "-k0", f"-j{cpu_count}", "-C", args.build_dir, "all"],
            env=new_env,
            check=True,
        )
