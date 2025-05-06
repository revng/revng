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
        parser.add_argument("--seed", help="Seed to use in the RNG for shuffling")
        parser.add_argument("input", help="Input directory")
        parser.add_argument("output", help="Output directory")
        parser.add_argument(
            "config_files", nargs="+", help="Additional .yml files to pass to test-configure"
        )

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
        if args.seed is not None:
            seed = args.seed
        else:
            seed = getrandbits(32)
        Random(seed).shuffle(file_list)

        data = {"sources": [{"members": file_list}]}
        binaries_yml = NamedTemporaryFile("w", suffix=".yml")
        yaml.safe_dump(data, binaries_yml)

        run(
            ("revng", "test-configure", "--install-path", args.input)
            + ("--destination", args.output, binaries_yml.name, *args.config_files),
            check=True,
        )

        if args.meta is not None:
            with open(args.meta) as f:
                data = yaml.safe_load(f)
        else:
            data = {}
        data["seed"] = seed
        with open(Path(args.output) / "meta.yml", "w") as f:
            yaml.safe_dump(data, f)
