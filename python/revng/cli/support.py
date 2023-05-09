#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shlex
import signal
import sys
from dataclasses import dataclass
from shutil import which
from subprocess import Popen
from typing import Any, Dict, Iterable, List, Mapping, NoReturn, Optional, Tuple, Union

from revng.support import get_root, read_lines
from revng.support.collect import collect_files, collect_libraries
from revng.support.elf import is_executable

additional_bin_paths = read_lines(get_root() / "share/revng/additional-bin-paths")
OptionalEnv = Optional[Mapping[str, str]]


@dataclass
class Options:
    parsed_args: Any
    remaining_args: List[str]
    command_prefix: List[str]
    verbose: bool
    dry_run: bool
    keep_temporaries: bool
    search_prefixes: List[str]


def shlex_join(split_command: Iterable[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in split_command)


def log_error(msg: str):
    sys.stderr.write(msg + "\n")


def wrap(args: List[str], command_prefix: List[str]):
    return command_prefix + args


def relative(path: str) -> str:
    relative_path = os.path.relpath(path, os.getcwd())
    if len(relative_path) < len(path):
        return relative_path
    else:
        return path


def _run_common(
    command, options: Options, environment: OptionalEnv = None
) -> Tuple[List[str], Dict[str, str]]:
    if not os.path.isfile(command[0]):
        command[0] = get_command(command[0], options.search_prefixes)

    if len(options.command_prefix) > 0:
        if is_executable(command[0]):
            command = wrap(command, options.command_prefix)
        else:
            sh = get_command("sh", options.search_prefixes)
            command = wrap([sh, "-c", 'exec "$0" "$@"', *command], options.command_prefix)

    if options.verbose:
        program_path = relative(command[0])
        sys.stderr.write("{}\n\n".format(" \\\n  ".join([program_path] + command[1:])))

    environment = dict(os.environ if environment is None else environment)

    if "valgrind" in command:
        environment["PYTHONMALLOC"] = "malloc"

    return command, environment


def popen(command, options: Options, environment: OptionalEnv = None, **kwargs) -> int | Popen:
    command, environment = _run_common(command, options, environment)

    if options.dry_run:
        return 0

    return Popen(command, env=environment, **kwargs)


def try_run(command, options: Options, environment: OptionalEnv = None) -> int:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        process = popen(
            command,
            options,
            environment,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
        )
        if isinstance(process, int):
            return process
        return process.wait()
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def run(command, options: Options, environment: OptionalEnv = None):
    result = try_run(command, options, environment)
    if result != 0:
        sys.exit(result)


def exec_run(command, options: Options, environment: OptionalEnv) -> Union[int, NoReturn]:
    command, environment = _run_common(command, options, environment)

    if options.dry_run:
        return 0

    return os.execvpe(command[0], command, environment)


def get_command(command: str, search_prefixes: Iterable[str]) -> str:
    for additional_bin_path in additional_bin_paths:
        for executable in collect_files(search_prefixes, [additional_bin_path], command):
            return executable

    path = which(command)
    if not path:
        log_error(f'Couldn\'t find "{command}".')
        assert False
    return os.path.abspath(path)


def interleave(base: List[str], repeat: str):
    return list(sum(zip([repeat] * len(base), base), ()))


def handle_asan(dependencies: Iterable[str], search_prefixes: Iterable[str]) -> List[str]:
    libasan = [name for name in dependencies if ("libasan." in name or "libclang_rt.asan" in name)]

    if len(libasan) != 1:
        return []

    libasan_path = relative(libasan[0])
    original_asan_options = os.environ.get("ASAN_OPTIONS", "")
    if original_asan_options:
        asan_options = dict([option.split("=") for option in original_asan_options.split(":")])
    else:
        asan_options = {}
    asan_options["abort_on_error"] = "1"
    asan_options["detect_leaks"] = "0"
    new_asan_options = ":".join(["=".join(option) for option in asan_options.items()])

    # Use `sh` instead of `env` since `env` sometimes is not a real executable
    # but a shebang script spawning /usr/bin/coreutils, which makes gdb unhappy
    return [
        get_command("sh", search_prefixes),
        "-c",
        f'ASAN_OPTIONS={new_asan_options} ld.so --preload {libasan_path} "$0" "$@"',
    ]


def build_command_with_loads(command: str, args: Iterable[str], options: Options) -> List[str]:
    (to_load, dependencies) = collect_libraries(options.search_prefixes)
    prefix = handle_asan(dependencies, options.search_prefixes)

    return (
        prefix
        + [relative(get_command(command, options.search_prefixes))]
        + interleave(to_load, "-load")
        + args
    )


def executable_name() -> str:
    return os.path.basename(sys.argv[0])
