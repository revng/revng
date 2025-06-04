#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import os
from pathlib import Path
from random import Random, getrandbits
from subprocess import run
from tempfile import NamedTemporaryFile

import yaml

from revng.internal.cli.commands_registry import Command, Options


class MassTestingConfigureCommand(Command):
    def __init__(self):
        super().__init__(
            ("mass-testing", "configure"), "Collect binary files and run test-configure"
        )

    def register_arguments(self, parser: argparse.ArgumentParser):
        parser.description = "Collect binary files and run test-configure"
        parser.add_argument("--meta", help="Metadata definition")
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument("config_file", help="Config file to be read by configure")

    def run(self, options: Options):
        args = options.parsed_args

        file_list = []
        input_path = Path(args.input)
        for dirpath, _, filenames in os.walk(args.input):
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

        with open(args.config_file) as f:
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
                ("revng", "test-configure", "--install-path", args.input)
                + ("--destination", args.output, config_yml.name),
                check=True,
            )

        data: dict = {}
        if args.meta is not None:
            with open(args.meta) as f:
                data = yaml.safe_load(f)
        data["seed"] = seed
        if sample is not None:
            data["sample"] = sample
        data["configurations"] = config
        with open(Path(args.output) / "meta.yml", "w") as f:
            yaml.safe_dump(data, f)
