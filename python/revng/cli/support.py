#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, NoReturn, Optional, Union

from revng.support.collect import collect_files, collect_libraries
from revng.support.elf import is_executable

try:
    from shutil import which
except ImportError:
    from backports.shutil_which import which  # type: ignore

script_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(script_path, "..", "..", "..", "..")


@dataclass
class Options:
    parsed_args: Any
    remaining_args: List[str]
    command_prefix: List[str]
    verbose: bool
    dry_run: bool
    keep_temporaries: bool
    search_prefixes: List[str] = field(default_factory=lambda: [root_path])


def shlex_join(split_command: Iterable[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in split_command)


def log_error(msg: str):
    sys.stderr.write(msg + "\n")


def wrap(args: List[str], command_prefix: List[str]):
    return command_prefix + args


def relative(path: str) -> str:
    return os.path.relpath(path, os.getcwd())


def try_run(
    command, options: Options, environment: Optional[Dict[str, str]] = None, do_exec=False
) -> Union[int, NoReturn]:
    if is_executable(command[0]):
        command = wrap(command, options.command_prefix)

    if options.verbose:
        program_path = relative(command[0])
        sys.stderr.write("{}\n\n".format(" \\\n  ".join([program_path] + command[1:])))

    if options.dry_run:
        return 0

    if environment is None:
        environment = dict(os.environ)

    if do_exec:
        return os.execvpe(command[0], command, environment)
    else:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        process = subprocess.Popen(
            command,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
            env=environment,
        )
        return_code = process.wait()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        return return_code


def run(command, options: Options, environment: Optional[Dict[str, str]] = None, do_exec=False):
    result = try_run(command, options, environment, do_exec)
    if result != 0:
        log_error(f"The following command exited with {result}:\n{shlex_join(command)}")
        sys.exit(result)


def get_command(command: str, search_prefixes: Iterable[str]) -> str:
    if command.startswith("revng-"):
        for executable in collect_files(search_prefixes, ["libexec", "revng"], command):
            return executable
        log_error(f'Couldn\'t find "{command}"')
        assert False

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
    new_asan_options = ":".join(["=".join(option) for option in asan_options.items()])

    # Use `sh` instead of `env` since `env` sometimes is not a real executable
    # but a shebang script spawning /usr/bin/coreutils, which makes gdb unhappy
    return [
        get_command("sh", search_prefixes),
        "-c",
        f'LD_PRELOAD={libasan_path} ASAN_OPTIONS={new_asan_options} exec "$0" "$@"',
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
