#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import time
from pathlib import Path
from subprocess import run

import click
import yaml

from revng.support import get_root


@click.command(name="run", help="Run a mass-testing configuration")
@click.option("-j", "--jobs", type=int, help="Number of parallel jobs to use")
@click.argument("build_dir", metavar="BUILD_DIR")
def mass_testing_run(jobs: int | None, build_dir: str) -> None:
    meta_file = Path(build_dir) / "meta.yml"
    with open(meta_file) as f:
        meta_data = yaml.safe_load(f)

    if "MASS_TESTING_RUN_JOBS" in os.environ:
        cpu_count = int(os.environ["MASS_TESTING_RUN_JOBS"])
    elif jobs:
        cpu_count = jobs
    else:
        cpu_count = len(os.sched_getaffinity(0))
        memory_limits = [
            x["memory_limit"] for x in meta_data["configurations"] if "memory_limit" in x
        ]
        if len(memory_limits) > 0:
            # If one or more configuration(s) set the memory limit, compute
            # the number of jobs so that each one, in the worst-case
            # scenario (biggest memory limit), has enough memory. Account
            # for a 10% of "system" memory that should not be used.
            max_ram = max(memory_limits)
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            cpu_count = min(int((mem_bytes * 0.9) // max_ram), cpu_count)

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
        ["ninja", "--quiet", "-k0", f"-j{cpu_count}", "-C", build_dir, "all"],
        env=new_env,
        check=True,
    )
