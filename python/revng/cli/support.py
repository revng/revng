#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from elftools.elf.dynamic import DynamicSegment
from elftools.elf.elffile import ELFFile

try:
    from shutil import which
except ImportError:
    from backports.shutil_which import which  # type: ignore


@dataclass
class Options:
    parsed_args: Any
    remaining_args: List[str]
    search_prefixes: List[str]
    command_prefix: List[str]
    verbose: bool
    dry_run: bool
    keep_temporaries: bool


def shlex_join(split_command: Iterable[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in split_command)


def log_error(msg: str):
    sys.stderr.write(msg + "\n")


def wrap(args: List[str], command_prefix: List[str]):
    return command_prefix + args


def relative(path: str) -> str:
    return os.path.relpath(path, os.getcwd())


def run(command, options: Options, environment: Optional[Dict[str, str]] = None):
    if is_executable(command[0]):
        command = wrap(command, options.command_prefix)

    if options.verbose:
        program_path = relative(command[0])
        sys.stderr.write("{}\n\n".format(" \\\n  ".join([program_path] + command[1:])))

    if options.dry_run:
        return

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if environment is None:
        environment = dict(os.environ)
    p = subprocess.Popen(
        command, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL), env=environment
    )
    if p.wait() != 0:
        log_error(f"The following command exited with {p.returncode}:\n{shlex_join(command)}")
        sys.exit(p.returncode)


def is_executable(path: str) -> bool:
    with open(path, "rb") as program:
        return program.read(4) == b"\x7fELF"


def is_dynamic(path: str) -> bool:
    with open(path, "rb") as input_file:
        return (
            len(
                [
                    segment
                    for segment in ELFFile(input_file).iter_segments()
                    if segment.header.p_type == "PT_DYNAMIC"
                ]
            )
            != 0
        )


def get_command(command: str, search_prefixes: Iterable[str]) -> str:
    if command.startswith("revng-"):
        for executable in collect_files(search_prefixes, ["libexec", "revng"], command):
            return executable
        log_error(f'Couldn\'t find "{command}"')
        assert False

    path = which(command)
    if not path:
        log_error('Couldn\'t find "{command}".')
        assert False
    return os.path.abspath(path)


def interleave(base: List[str], repeat: str):
    return list(sum(zip([repeat] * len(base), base), ()))


# Use in case different version of pyelftools might give str or bytes
def to_string(obj: Union[str, bytes]) -> str:
    if isinstance(obj, str):
        return obj
    return obj.decode("utf-8")


def get_elf_needed(path: str) -> List[str]:
    with open(path, "rb") as elf_file:
        segments = [
            segment
            for segment in ELFFile(elf_file).iter_segments()
            if isinstance(segment, DynamicSegment)
        ]

        if len(segments) != 1:
            return []

        needed = [
            to_string(tag.needed)
            for tag in segments[0].iter_tags()
            if tag.entry.d_tag == "DT_NEEDED"
        ]

        elf_runpath: List[str] = [
            tag.runpath for tag in segments[0].iter_tags() if tag.entry.d_tag == "DT_RUNPATH"
        ]

        assert len(elf_runpath) < 2

        if not elf_runpath:
            return needed

        runpaths = [
            runpath.replace("$ORIGIN", str(Path(path).parent))
            for runpath in elf_runpath[0].split(":")
        ]
        absolute_needed = []
        for lib in needed:
            found = False
            for runpath in runpaths:
                full_path = Path(runpath) / lib
                if full_path.is_file():
                    absolute_needed.append(str(full_path.resolve()))
                    found = True
                    break

            if not found:
                absolute_needed.append(lib)

        return absolute_needed


def collect_files(
    search_prefixes: Iterable[str], path_components: Iterable[str], pattern: str
) -> List[str]:
    to_load: Dict[str, Path] = {}
    for prefix in search_prefixes:
        analyses_path = Path(prefix).joinpath(*path_components)
        if not analyses_path.is_dir():
            continue

        # Enumerate all the libraries containing analyses
        for library in analyses_path.glob(pattern):
            if library.name not in to_load:
                to_load[library.name] = library.resolve()

    return [str(v) for v in to_load.values()]


def collect_libraries(search_prefixes: Iterable[str]) -> Tuple[List[str], Set[str]]:
    to_load = collect_files(search_prefixes, ["lib", "revng", "analyses"], "*.so")

    # Identify all the libraries that are dependencies of other libraries, i.e.,
    # non-roots in the dependencies tree. Note that circular dependencies are
    # not allowed.
    dependencies = set(chain.from_iterable([get_elf_needed(path) for path in to_load]))
    return (to_load, dependencies)


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
