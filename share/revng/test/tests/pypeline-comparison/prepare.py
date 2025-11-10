#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import sys
from hashlib import file_digest
from pathlib import Path
from shutil import copy
from subprocess import run
from tempfile import TemporaryFile

import yaml


# This script prepares the output directory for `revng project ...` commands to
# be run. Eventually it will be replaced by `revng project init` once that's
# working.
def main():
    input_binary = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    with TemporaryFile("wb+") as temp_model:
        # Generate a standard model file from the binary, via initial auto-analysis
        run(["revng", "analyze", "revng-initial-auto-analysis", input_binary], stdout=temp_model)
        temp_model.seek(0)
        model = yaml.safe_load(temp_model)

    with open(input_binary, "rb") as f:
        hash_ = file_digest(f, "sha256").hexdigest()
        size = f.seek(0, os.SEEK_END)

    # Patch the `Binaries` entry with the input file
    model["Binaries"] = [{"Index": 0, "Hash": hash_, "Size": size, "Name": "input"}]
    # Add a reference to the newly-added binary to all the segments
    for segment in model["Segments"]:
        segment["Binary"] = "/Binaries/0"

    with open(output_dir / "model.yml", "w") as model_out:
        yaml.safe_dump(model, model_out)

    copy(input_binary, output_dir / "input")


if __name__ == "__main__":
    main()
