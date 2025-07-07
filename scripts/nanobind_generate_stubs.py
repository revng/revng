#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import importlib
import os
import sys
from ctypes import CDLL

from nanobind.stubgen import StubGen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output file (stdout if omitted)")
    parser.add_argument(
        "-m", "--module", required=True, help="Nanobind module to generate annotations from"
    )
    parser.add_argument(
        "-l",
        "--library",
        default=[],
        action="append",
        help="Shared library to preload before generating stubs",
    )
    args = parser.parse_args()

    handles = []
    for path in args.library:
        handles.append(CDLL(path, os.RTLD_NOW | os.RTLD_GLOBAL))

    target_module = importlib.import_module(args.module)
    stub_gen = StubGen(target_module, quiet=False)
    stub_gen.put(target_module)
    stub_text = stub_gen.get()

    if args.output is None:
        sys.stdout.write(stub_text)
        sys.stdout.flush()
    else:
        with open(args.output, "w") as f:
            f.write(stub_text)


if __name__ == "__main__":
    main()
