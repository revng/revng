#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import mmap
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from shutil import which
from typing import IO, Generator, Iterable, List, Optional, TypeVar, Union, cast

from llvmcpy import LLVMCPy

T = TypeVar("T")
AnyPath = Union[str, Path]
SingleOrIterable = Union[T, Iterable[T]]
AnyPaths = SingleOrIterable[AnyPath]


def log_error(msg: str):
    sys.stderr.write(msg + "\n")


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text().strip().split("\n")


# We assume this file is in <root>/lib/python$VERSION/site-packages/revng/
def get_root() -> Path:
    return (Path(__file__) / "../../../../../../").resolve()


additional_bin_paths = read_lines(get_root() / "share/revng/additional-bin-paths")


def collect_files_gen(
    search_prefixes: AnyPaths, path_components: Iterable[AnyPath], pattern: str
) -> Generator[str, None, None]:
    if isinstance(search_prefixes, (str, Path)):
        prefixes: Iterable[AnyPath] = (search_prefixes,)
    else:
        prefixes = search_prefixes

    already_found = set()
    for prefix in prefixes:
        analyses_path = Path(prefix).joinpath(*path_components)
        if not analyses_path.is_dir():
            continue

        # Enumerate all the libraries containing analyses
        for library in analyses_path.glob(pattern):
            if library.is_file() and library.name not in already_found:
                already_found.add(library.name)
                yield str(library.resolve())


def collect_files(
    search_prefixes: AnyPaths, path_components: Iterable[AnyPath], pattern: str
) -> List[str]:
    return list(collect_files_gen(search_prefixes, path_components, pattern))


def collect_one(
    search_prefixes: AnyPaths, path_components: Iterable[AnyPath], pattern: str
) -> Optional[str]:
    return next(collect_files_gen(search_prefixes, path_components, pattern), None)


def _get_command(command: str, search_prefixes: Iterable[str]) -> Optional[str]:
    for additional_bin_path in additional_bin_paths:
        for executable in collect_files(search_prefixes, [additional_bin_path], command):
            return executable

    path = which(command)
    if path is not None:
        return os.path.abspath(path)
    return None


def get_command(command: str, search_prefixes: Iterable[str]) -> str:
    path = _get_command(command, search_prefixes)
    if path is None:
        log_error(f'Couldn\'t find "{command}".')
        raise ValueError()
    return path


def get_llvmcpy():
    return LLVMCPy(_get_command("llvm-config", [get_root()]))


ToBytesInput = Union[str, bytes, IO[str], IO[bytes], mmap.mmap]


@contextmanager
def to_bytes(input_: ToBytesInput) -> Generator[bytes, None, None]:
    """Wrapper that allows to seamlessly treat strings, bytes and files as raw
    bytes. Strings will be encoded in utf-8. Files will be mmap-ed if possible
    otherwise their contents will be read in memory"""
    if isinstance(input_, mmap.mmap):
        yield cast(bytes, input_)
    elif not isinstance(input_, (str, bytes)):
        # Test to exclude things like stdin
        if input_.seekable():
            offset = input_.tell()
            with mmap.mmap(input_.fileno(), 0, access=mmap.ACCESS_READ, offset=offset) as mm:
                yield cast(bytes, mm)
        else:
            result = input_.read()
            if isinstance(result, str):
                yield result.encode("utf-8")
            else:
                yield result
    else:
        if isinstance(input_, str):
            yield input_.encode("utf-8")
        else:
            yield input_


def get_example_binary_path() -> str:
    runtime_dir = get_root() / "share/revng/test/tests/runtime"
    for entry in runtime_dir.iterdir():
        if not entry.is_file():
            continue
        if re.match(r"^calc-x86-64-static-revng-qa\.compiled-[0-9a-f]{8}$", entry.name):
            return str(entry.resolve())
    raise ValueError("Could not find calc binary")
