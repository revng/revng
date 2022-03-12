from os import path
from .support import run
from .revng import run_revng_command


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


def run_translate(args, post_dash_dash_args, search_prefixes, command_prefix):
    out_file = args.output if args.output else args.input[0] + ".translated"

    step_name = "EndRecompile"
    if args.isolate:
        step_name = step_name + "Isolated"

    command = [
        "revng",
        "pipeline",
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

    command += post_dash_dash_args

    return run_revng_command(command, search_prefixes, command_prefix)
