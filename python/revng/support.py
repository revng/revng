#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import os
import sys
from pathlib import Path
from shutil import which
from typing import Generator, Iterable, List, Optional, TypeVar, Union

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
