#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import os
import shlex
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Iterable, List, Literal

import click
import yaml

from revng.internal.support.collect import collect_libraries
from revng.support import TarDictionary, get_command, get_root, read_lines


@dataclass
class Options:
    parsed_args: Any
    remaining_args: List[str]
    command_prefix: List[str]
    verbose: bool
    dry_run: bool
    keep_temporaries: bool
    search_prefixes: List[str]


def search_prefixes() -> List[str]:
    """
    The prefixes where the programs and libraries shipped with revng are looked
    up: the revng root, which takes precedence, followed by the prefixes listed
    in `additional-search-prefixes`, if any.
    """
    prefixes = [str(get_root())]
    additional = read_lines(get_root() / "additional-search-prefixes")
    prefixes.extend(prefix for prefix in additional if prefix not in prefixes)
    return prefixes


def shlex_join(split_command: Iterable[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in split_command)


def relative(path: str) -> str:
    relative_path = os.path.relpath(path, os.getcwd())
    if len(relative_path) < len(path):
        return relative_path
    else:
        return path


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


def build_command_with_loads(command: str, args: Iterable[str]) -> List[str]:
    prefixes = search_prefixes()
    (to_load, dependencies) = collect_libraries(prefixes)
    prefix = handle_asan(dependencies, prefixes)

    return (
        prefix
        + [relative(get_command(command, prefixes))]
        + interleave(to_load, "-load")
        + list(args)
    )


def executable_name() -> str:
    return os.path.basename(sys.argv[0])


def to_string(filename: str, raw: bytes) -> str:
    return raw.decode("utf8")


def extract_tar[T](raw: bytes, process: Callable[[str, bytes], Any] = to_string) -> Dict[str, Any]:
    return {key: process(key, value) for key, value in TarDictionary(raw).items()}


def to_yaml(filename: str, raw: bytes) -> str:
    return yaml.safe_load(raw)


def is_file_executable(filename: str) -> bool:
    stat = os.stat(filename)
    return stat.st_mode & 0o111 == 0o111


def temporary_file_gen(prefix: str, keep_temporaries: bool):
    def temporary_file(suffix="", mode="w+"):
        return NamedTemporaryFile(
            prefix=prefix,
            suffix=suffix,
            mode=mode,
            delete=not keep_temporaries,
        )

    return temporary_file


keep_temporaries_option = click.option(
    "--keep-temporaries", is_flag=True, help="Do not delete temporary files."
)


@contextmanager
def file_wrapper(path: str | None, mode: Literal["r", "rb", "w", "wb"]):
    """Automatic wrapper for `open` for command-line arguments, to be used in
    a `with` statement. Will automatically return stdin/stdout based on `mode`
    in case the provided path is None or "-".
    """
    assert mode in ("r", "rb", "w", "wb")
    if path is None or path == "-":
        if "r" in mode:
            yield sys.stdin.buffer if "b" in mode else sys.stdin
        else:
            yield sys.stdout.buffer if "b" in mode else sys.stdout
    else:
        with open(path, mode) as f:
            yield f
