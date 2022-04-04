#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys
import os
import glob
import signal
import subprocess
import shlex

from itertools import chain
from dataclasses import dataclass
from typing import List, Any

from elftools.elf.dynamic import DynamicSegment
from elftools.elf.elffile import ELFFile

try:
    from shutil import which
except ImportError:
    from backports.shutil_which import which


@dataclass
class Options:
    parsed_args: Any
    remaining_args: List[str]
    search_prefixes: List[str]
    command_prefix: List[str]
    verbose: bool
    dry_run: bool
    keep_temporaries: bool


def shlex_join(split_command):
    return " ".join(shlex.quote(arg) for arg in split_command)


def log_error(msg):
    sys.stderr.write(msg + "\n")


def relative(path):
    return os.path.relpath(path, ".")


def wrap(args, command_prefix):
    return command_prefix + args


def relative(path: str):
    return os.path.relpath(path, os.getcwd())


def run(command, options: Options):
    if is_executable(command[0]):
        command = wrap(command, options.command_prefix)

    if options.verbose:
        program_path = relative(command[0])
        sys.stderr.write("{}\n\n".format(" \\\n  ".join([program_path] + command[1:])))

    if options.dry_run:
        return 0

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    environment = dict(os.environ)
    p = subprocess.Popen(
        command, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL), env=environment
    )
    if p.wait() != 0:
        log_error(f"The following command exited with {p.returncode}:\n{shlex_join(command)}")
        sys.exit(p.returncode)


def is_executable(path):
    with open(path, "rb") as program:
        return program.read(4) == b"\x7fELF"


def is_dynamic(path):
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


def get_command(command, search_prefixes):
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


def interleave(base, repeat):
    return list(sum(zip([repeat] * len(base), base), ()))


# Use in case different version of pyelftools might give str or bytes


def to_string(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, bytes):
        return obj.decode("utf-8")


def get_elf_needed(path):
    with open(path, "rb") as elf_file:
        segments = [
            segment
            for segment in ELFFile(elf_file).iter_segments()
            if isinstance(segment, DynamicSegment)
        ]

        if len(segments) == 1:
            needed = [
                to_string(tag.needed)
                for tag in segments[0].iter_tags()
                if tag.entry.d_tag == "DT_NEEDED"
            ]

            runpath = [
                tag.runpath for tag in segments[0].iter_tags() if tag.entry.d_tag == "DT_RUNPATH"
            ]

            assert len(runpath) < 2

            if not runpath:
                return needed
            else:
                runpaths = [
                    runpath.replace("$ORIGIN", os.path.dirname(path))
                    for runpath in runpath[0].split(":")
                ]
                absolute_needed = []
                for lib in needed:
                    found = False
                    for runpath in runpaths:
                        full_path = os.path.join(runpath, lib)
                        if os.path.isfile(full_path):
                            absolute_needed.append(full_path)
                            found = True
                            break

                    if not found:
                        absolute_needed.append(lib)

                return absolute_needed

        else:
            return []


def collect_files(search_prefixes, path_components, pattern):
    to_load = {}
    for prefix in search_prefixes:
        analyses_path = os.path.join(prefix, *path_components)
        if not os.path.isdir(analyses_path):
            continue

        # Enumerate all the libraries containing analyses
        for library in glob.glob(os.path.join(analyses_path, pattern)):
            basename = os.path.basename(library)
            if basename not in to_load:
                to_load[basename] = relative(library)

    return list(to_load.values())


def collect_libraries(search_prefixes):
    to_load = collect_files(search_prefixes, ["lib", "revng", "analyses"], "*.so")

    # Identify all the libraries that are dependencies of other libraries, i.e.,
    # non-roots in the dependencies tree. Note that circular dependencies are
    # not allowed.
    dependencies = set(chain.from_iterable([get_elf_needed(path) for path in to_load]))
    return (to_load, dependencies)


def handle_asan(dependencies, search_prefixes):
    libasan = [name for name in dependencies if ("libasan." in name or "libclang_rt.asan" in name)]

    if len(libasan) != 1:
        return []

    libasan_path = relative(libasan[0])
    original_asan_options = os.environ.get("ASAN_OPTIONS", "")
    if original_asan_options:
        asan_options = dict([option.split("=") for option in original_asan_options.split(":")])
    else:
        asan_options = dict()
    asan_options["abort_on_error"] = "1"
    new_asan_options = ":".join(["=".join(option) for option in asan_options.items()])

    # Use `sh` instead of `env` since `env` sometimes is not a real executable
    # but a shebang script spawning /usr/bin/coreutils, which makes gdb unhappy
    return [
        get_command("sh", search_prefixes),
        "-c",
        f'LD_PRELOAD={libasan_path} ASAN_OPTIONS={new_asan_options} exec "$0" "$@"',
    ]


def build_command_with_loads(command, args, options):
    (to_load, dependencies) = collect_libraries(options.search_prefixes)
    prefix = handle_asan(dependencies, options.search_prefixes)

    return (
        prefix
        + [relative(get_command(command, options.search_prefixes))]
        + interleave(to_load, "-load")
        + args
    )
