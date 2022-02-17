#!/usr/bin/env python3

import sys
import argparse
import signal
import os
import subprocess
from os import path
from revng_support import run, get_command, build_command_with_loads


def register_translate(subparsers):
    parser = subparsers.add_parser("translate", description="revng translate wrapper")
    parser.add_argument("input", type=str, nargs=1, help="Input binary to be translated")

    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT",
        dest="output",
        type=str,
        nargs="?",
        help="Translated output",
    )

    parser.add_argument("-i", "--isolate", action="store_true", help="Enable function isolation.")

    parser.add_argument(
        "--trace", action="store_true", help="Use the tracing version of support.ll."
    )

    parser.add_argument("-O0", action="store_true", help="Do no optimize.")
    parser.add_argument("-O1", action="store_true", help="Use llc -O2.")
    parser.add_argument("-O2", action="store_true", help="Use llc -O2 and opt -O2.")

    parser.add_argument("--base", help="Load address to employ in lifting.")


def run_translate(args, post_dash_dash_args, search_path, search_prefixes, command_prefix):
    main_path = path.abspath(__file__)
    main_path = path.dirname(main_path)

    command = []
    command.append("-P")
    command.append(main_path + "/pipelines/translate.yml")

    command.append("-P")
    command.append(main_path + "/pipelines/isolate-translate.yml")

    out_file = args.output if args.output else args.input[0] + ".translated"

    step_name = "EndRecompile"
    if args.isolate:
        step_name = step_name + "Isolated"
    command = command + [
        "--silent",
        "output:root:Translated",
        "--step",
        step_name,
        "-i",
        "Lift:input:" + args.input[0],
        "-o",
        step_name + ":output:" + out_file,
    ]

    if args.O1:
        command.append("--compile-opt-level=1")
    elif args.O2:
        command += ["--compile-opt-level=2", "-f", "O2"]
    else:
        command.append("--compile-opt-level=0")

    if args.base:
        command.append(f"--base={args.base}")

    if args.trace:
        command.append("--link-trace")

    to_execute = build_command_with_loads("revng-pipeline", command, search_path, search_prefixes)

    return run(to_execute + post_dash_dash_args, command_prefix)


def main():
    parser = argparse.ArgumentParser(description="The rev.ng translator.")
    subparsers = parser.add_subparsers(dest="command_name", help="sub-commands help")
    register_translate(subparsers)
    args = parser.parse_args()
    return run_translate(args, [], [], [])


if __name__ == "__main__":
    sys.exit(main())
