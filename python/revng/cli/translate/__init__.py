import sys
import argparse
import signal
import os
import subprocess
from os import path
from ..support import run, get_command


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
    out_file = args.output if args.output else args.input[0] + ".translated"

    step_name = "EndRecompile"
    if args.isolate:
        step_name = step_name + "Isolated"
    command = [
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

    return run(
        [get_command("revng", search_path), "pipeline"] + command + post_dash_dash_args,
        command_prefix,
    )


# WIP: outline
def register_lift(subparsers):
    parser = subparsers.add_parser("lift", description="revng lift wrapper")
    parser.add_argument("input", type=str, nargs=1, help="Input binary to be lifted")
    parser.add_argument("output", type=str, nargs=1, help="Output module path")
    parser.add_argument("--record-asm", action="store_true")
    parser.add_argument("--record-ptc", action="store_true")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--base", type=str)
    parser.add_argument("--entry", type=str)
    parser.add_argument("--debug-info", type=str)
    parser.add_argument("--import-debug-info", type=str, action="append", default=[])


def run_lift(args, post_dash_dash_args, search_path, search_prefixes, command_prefix):
    # Run revng model import args.input
    arg_or_empty = (
        lambda args, name: [f"--{name}={args.__dict__[name]}"] if args.__dict__[name] else []
    )

    # WIP
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix=".yml") as model, NamedTemporaryFile(
        suffix=".ll"
    ) as model_in_module:
        run(
            (
                [
                    get_command("revng-model-import-binary", search_path),
                    args.input[0],
                    "-o",
                    model.name,
                ]
                + [f"--import-debug-info={value}" for value in args.import_debug_info]
                + arg_or_empty(args, "base")
                + arg_or_empty(args, "entry")
            ),
            command_prefix,
        )

        run(
            [
                get_command("revng-model-inject", search_path),
                model.name,
                "/dev/null",
                "-o",
                model_in_module.name,
            ],
            command_prefix,
        )

        run(
            (
                [
                    get_command("revng", search_path),
                    "opt",
                    model_in_module.name,
                    "-o",
                    args.output[0],
                    "-load-binary",
                    f"-binary-path={args.input[0]}",
                    "-lift",
                ]
                + arg_or_empty(args, "external")
                + arg_or_empty(args, "record_asm")
                + arg_or_empty(args, "record_ptc")
            ),
            command_prefix,
        )

    # WIP NEXT: debug-info
