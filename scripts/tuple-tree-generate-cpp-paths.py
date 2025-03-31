#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from argparse import ArgumentParser, Namespace
from typing import List

import yaml


def parse_args() -> Namespace:
    argparser = ArgumentParser()
    argparser.add_argument("schema", help="Path to YAML schema")
    argparser.add_argument("output_dir", help="Output to this directory")
    argparser.add_argument(
        "--forward-decls",
        action="store_true",
        default=False,
        help="If set, prints the forward declarations path",
    )
    argparser.add_argument(
        "--early", action="store_true", help="If set, prints the early headers paths"
    )
    argparser.add_argument(
        "--late", action="store_true", help="If set, prints the late headers paths"
    )
    argparser.add_argument(
        "--impl", action="store_true", help="If set, prints the implementation path"
    )

    return argparser.parse_args()


def main():
    args = parse_args()

    with open(args.schema, encoding="utf-8") as f:
        raw_schema = yaml.safe_load(f)

    paths: List[str] = []

    if args.forward_decls:
        paths.append(f"{args.output_dir}/ForwardDecls.h")

    if args.early:
        for definition in raw_schema["definitions"]:
            paths.append(f"{args.output_dir}/Early/{definition['name']}.h")

    if args.late:
        for definition in raw_schema["definitions"]:
            paths.append(f"{args.output_dir}/Late/{definition['name']}.h")

    if args.impl:
        paths.append(f"{args.output_dir}/Impl.cpp")

    print("\n".join(paths), end="")


if __name__ == "__main__":
    main()
