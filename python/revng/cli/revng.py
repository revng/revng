import argparse
import os
import subprocess
import sys

try:
    from shutil import which
except ImportError:
    from backports.shutil_which import which

from .support import (
    build_command_with_loads,
    set_verbose,
    get_command,
    run,
    log_error,
    interleave,
    collect_files,
)

search_prefixes = []
real_argv0 = os.environ.get("REAL_ARGV0", sys.argv[0])


def split_dash_dash(args):
    if not args:
        return [], []

    extra_args = []
    while (len(args) != 0) and (args[0] != "--"):
        extra_args.append(args[0])
        del args[0]

    # Drop the delimiter
    if len(args) != 0:
        del args[0]

    return list(extra_args), args


def find_file(name, paths):
    for path in paths:
        path = os.path.join(path, name)
        if os.path.isfile(path):
            return path

    log_error("Can't find the following file: {}".format(name))
    assert False


def get_stderr(args):
    with subprocess.Popen(args, stderr=subprocess.PIPE) as process:
        return process.stderr.read()


def run_pipeline(args, search_prefixes, command_prefix):
    pipelines = collect_files(search_prefixes, ["share", "revng", "pipelines"], "*.yml")
    pipelines_args = interleave(pipelines, "-P")
    command = build_command_with_loads("revng-pipeline", pipelines_args + args, search_prefixes)
    return run(command, command_prefix)


def run_cc(args, search_prefixes, command_prefix):
    # Collect translate options
    translated_args = []
    if "--" in args:
        translate_args, args = split_dash_dash(args)

    if (not args) or ("--help" in args):
        log_error(
            f"Usage: {real_argv0} cc [[translate-options] --] compiler " + "[compiler-options]"
        )
        return -1

    res = subprocess.call(args)
    if res != 0:
        return res

    # Are we linking?
    if not ("-c" in args):
        assert "-o" in args

        # Identify the path of the final program
        output = os.path.abspath(args[args.index("-o") + 1])

        original = output + ".original"
        translated = original + ".translated"
        os.rename(output, original)

        result = run_translate(translate_args + [original], search_prefixes, command_prefix)
        if result != 0:
            return result

        os.rename(translated, output)

    return res


def main():
    return run_revng_command(sys.argv, [], [])


def run_revng_command(arguments, search_prefixes, command_prefix):
    parser = argparse.ArgumentParser(description="The rev.ng driver.")
    parser.add_argument("--version", action="store_true", help="Display version information.")
    parser.add_argument("--verbose", action="store_true", help="Log all executed commands.")
    parser.add_argument(
        "--perf", action="store_true", help="Run programs under perf (for use with hotspot)."
    )
    parser.add_argument("--heaptrack", action="store_true", help="Run programs under heaptrack.")
    parser.add_argument("--gdb", action="store_true", help="Run programs under gdb.")
    parser.add_argument("--lldb", action="store_true", help="Run programs under lldb.")
    parser.add_argument("--valgrind", action="store_true", help="Run programs under valgrind.")
    parser.add_argument("--callgrind", action="store_true", help="Run programs under callgrind.")
    parser.add_argument(
        "--prefix", action="append", metavar="PREFIX", help="Additional search prefix."
    )

    # Get script_path
    global script_path
    script_path = os.path.dirname(os.path.realpath(__file__))

    subparsers = parser.add_subparsers(dest="command_name", help="sub-commands help")

    from .translate import register_translate
    from .lift import register_lift

    register_translate(subparsers)
    register_lift(subparsers)

    subparsers.add_parser("cc", help="compile, link and translate transparently", add_help=False)
    subparsers.add_parser("opt", help="LLVM's opt with rev.ng passes", add_help=False)
    subparsers.add_parser("model", help="Model commands", add_help=False)

    if len(search_prefixes) == 0:
        prefixes = []
        for arg, next in zip(arguments, arguments[1:] + [""]):
            if arg.startswith("--prefix="):
                prefixes += arg[len("--prefix=") :]
            elif arg == "--prefix" and next != "":
                prefixes += next
        search_prefixes = prefixes + [os.path.join(script_path, "..", "..", "..", "..")]

    prefix = "revng-"
    for executable in collect_files(search_prefixes, ["libexec", "revng"], f"{prefix}*"):
        name = os.path.basename(executable)[len(prefix) :]
        subparsers.add_parser(name, help=f"see {name} --help", add_help=False)

    global revng_parser
    revng_parser = parser

    # Strip away arguments -- so we can forward them to revng-lift
    base_args, post_dash_dash = split_dash_dash(arguments[1:])

    args, unknown_args = parser.parse_known_args(base_args)

    if args.verbose:
        set_verbose()

    if len(command_prefix) == 0:
        assert (args.gdb + args.lldb + args.valgrind + args.callgrind) <= 1

        if args.gdb:
            command_prefix += ["gdb", "-q", "--args"]

        if args.lldb:
            command_prefix += ["lldb", "--"]

        if args.valgrind:
            command_prefix += ["valgrind"]

        if args.callgrind:
            command_prefix += ["valgrind", "--tool=callgrind"]

        if args.perf:
            command_prefix += ["perf", "record", "--call-graph", "dwarf", "--output=perf.data"]

        if args.heaptrack:
            command_prefix += ["heaptrack"]

    if args.version:
        sys.stdout.write("rev.ng version @VERSION@\n")
        return 0

    return run_command(args, unknown_args, post_dash_dash, search_prefixes, command_prefix)


def run_subcommand(command, all_args, search_prefixes, command_prefix):
    target_executable = "revng-" + command
    for executable in collect_files(search_prefixes, ["libexec", "revng"], f"revng-*"):
        if executable.endswith("/" + target_executable):
            arguments = [executable] + all_args
            return run(arguments, command_prefix)

    log_error("Can't find the following command: {}".format(command))
    return 1


def run_command(args, unknown_args, post_dash_dash, search_prefixes, command_prefix):
    from .translate import run_translate
    from .lift import run_lift

    command = args.command_name
    if not command:
        log_error("No command specified")
        return 1

    all_args = list(unknown_args)
    if post_dash_dash:
        all_args += ["--"] + post_dash_dash

    # First consider the hardcoded commands
    if command == "opt":
        opt_command = build_command_with_loads(
            "opt", all_args + ["-serialize-model"], search_prefixes
        )
        run(opt_command, command_prefix)
    elif command == "cc":
        return run_cc(all_args, search_prefixes, command_prefix)
    elif command == "translate":
        return run_translate(args, post_dash_dash, search_prefixes, command_prefix)
    elif command == "lift":
        return run_lift(args, post_dash_dash, search_prefixes, command_prefix)
    elif command == "pipeline":
        return run_pipeline(all_args, search_prefixes, command_prefix)
    elif command == "model":
        if not all_args:
            log_error("No subcommand specified")
            return 1
        command += "-" + all_args[0]
        del all_args[0]

        if command == "model-import":
            if not all_args:
                log_error("No subcommand specified")
                return 1
            command += "-" + all_args[0]
            del all_args[0]

        return run_subcommand(command, all_args, search_prefixes, command_prefix)
    else:
        return run_subcommand(command, all_args, search_prefixes, command_prefix)
