#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import os
from fnmatch import fnmatch
from itertools import chain
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Set, Tuple

from . import AnyPath, AnyPaths
from .elf import get_elf_needed


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


def collect_files_recursive(
    search_prefixes: AnyPaths, path_components: Iterable[AnyPath], pattern: str
) -> Generator[Tuple[str, str], None, None]:
    # Recursive version of collect_files_gen, will go through `search_prefixes`
    # and look recursively for `pattern`. Returns a generator of a 2-tuple
    # consisting of the path relative to the search_prefix and the absolute path
    if isinstance(search_prefixes, (str, Path)):
        prefixes: Iterable[AnyPath] = (search_prefixes,)
    else:
        prefixes = search_prefixes

    already_found = set()
    for prefix in prefixes:
        # Add the path components to the prefix, skip if it's not a directory
        root_path = Path(prefix).joinpath(*path_components)
        if not root_path.is_dir():
            continue

        for dirname, directories, filenames in os.walk(root_path):
            dirname_path = Path(dirname)
            for filename in filenames:
                file_path = dirname_path / filename
                # Check that that file_path is an actual file (could be a
                # symlink, pipe, etc.) and that it matches the glob pattern
                # via fnmatch
                if not (file_path.is_file() and fnmatch(filename, pattern)):
                    continue

                file_path_relative = file_path.relative_to(root_path)
                if file_path_relative in already_found:
                    continue

                already_found.add(file_path_relative)
                yield (str(file_path_relative), str(file_path.resolve()))


def collect_files(
    search_prefixes: AnyPaths, path_components: Iterable[AnyPath], pattern: str
) -> List[str]:
    return list(collect_files_gen(search_prefixes, path_components, pattern))


def collect_one(
    search_prefixes: AnyPaths, path_components: Iterable[AnyPath], pattern: str
) -> Optional[str]:
    return next(collect_files_gen(search_prefixes, path_components, pattern), None)


def collect_libraries(search_prefixes: AnyPaths) -> Tuple[List[str], Set[str]]:
    to_load = collect_files(search_prefixes, ["lib", "revng", "analyses"], "*.so")

    # Identify all the libraries that are dependencies of other libraries, i.e.,
    # non-roots in the dependencies tree. Note that circular dependencies are
    # not allowed.
    dependencies = set(chain.from_iterable([get_elf_needed(path) for path in to_load]))
    return (to_load, dependencies)


def collect_pipelines(search_prefixes: AnyPaths) -> List[str]:
    return collect_files(search_prefixes, ["share", "revng", "pipelines"], "*.yml")
