#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import argparse
import sys

from nanobind.stubgen import StubGen

from revng.internal.support import import_pipebox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output file (stdout if omitted)")
    parser.add_argument(
        "-l",
        "--library",
        default=[],
        action="append",
        help="Shared library to preload before generating stubs",
    )
    args = parser.parse_args()

    module, handles = import_pipebox(args.library)
    stub_gen = StubGen(module, quiet=False)
    stub_gen.put(module)
    stub_text = stub_gen.get()

    if args.output is None:
        sys.stdout.write(stub_text)
        sys.stdout.flush()
    else:
        with open(args.output, "w") as f:
            f.write(stub_text)


if __name__ == "__main__":
    main()
