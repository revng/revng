#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from pathlib import Path
from random import Random, getrandbits
from subprocess import run
from tempfile import NamedTemporaryFile

import click
import yaml


@click.command(name="configure", help="Collect binary files and run test-configure")
@click.option("--meta", help="Metadata definition")
@click.argument("input_", metavar="INPUT")
@click.argument("output", metavar="OUTPUT")
@click.argument("config_file", metavar="CONFIG_FILE")
def configure(meta: str | None, input_: str, output: str, config_file: str) -> None:
    file_list = []
    input_path = Path(input_)
    for dirpath, _, filenames in os.walk(input_):
        dirpath_path = Path(dirpath)
        for filename in filenames:
            file_list.append(str((dirpath_path / filename).relative_to(input_path)))

    # Shuffle the input files, this avoids having a limited variety of
    # test runs for a partial run
    seed: str | int | None = os.environ.get("MASS_TESTING_CONFIGURE_SEED")
    if seed is None:
        seed = getrandbits(32)
    random = Random(seed)

    sample_raw = os.environ.get("MASS_TESTING_CONFIGURE_SAMPLE")
    if sample_raw is not None:
        sample = int(sample_raw)
        file_list = random.sample(file_list, sample)
    else:
        sample = None
        # Shuffle the input files, this avoids having a limited variety of
        # test runs for a partial run
        random.shuffle(file_list)

    with open(config_file) as f:
        config = yaml.safe_load(f)

    commands = []
    for entry in config:
        cmd_data = {"type": entry["name"], "from": [{"type": "source"}], "suffix": "/"}
        cmd = 'test-harness "$INPUT" "$OUTPUT"'
        if "timeout" in entry:
            cmd += f' --timeout {entry["timeout"]}'
        if "memory_limit" in entry:
            cmd += f' --memory-limit {entry["memory_limit"]}'
        cmd += f' -- {entry["command"]}'
        cmd_data["command"] = cmd
        commands.append(cmd_data)

    configure_data = {"sources": [{"members": file_list}], "commands": commands}
    with NamedTemporaryFile("w", suffix=".yml") as config_yml:
        yaml.safe_dump(configure_data, config_yml)
        run(
            ("revng", "test-configure", "--install-path", input_)
            + ("--destination", output, config_yml.name),
            check=True,
        )

    data: dict = {}
    if meta is not None:
        with open(meta) as f:
            data = yaml.safe_load(f)
    data["seed"] = seed
    if sample is not None:
        data["sample"] = sample
    data["configurations"] = config
    with open(Path(output) / "meta.yml", "w") as f:
        yaml.safe_dump(data, f)
