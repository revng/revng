#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
import sys
from typing import Any, Callable, Dict, List, Optional, Union, overload

import yaml

from revng.internal.cli.support import extract_tar, is_tar
from revng.ptml.console import Console, StringConsole
from revng.ptml.text import parse_ptml_plain, parse_ptml_yaml


def log(msg: str):
    sys.stderr.write(f"{msg}\n")


def yaml_load(content: str) -> Dict[str, Any]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(content, loader)


def normalize_filter_extract(filters: List[str], extract: Optional[str]) -> Union[str, List[str]]:
    if extract is not None:
        return extract
    if len(filters) == 0:
        return []
    return ",".join(filters).split(",")


def is_ptml(content: str) -> bool:
    return bool(re.match(r"\s*<", content))


def handle_multiple(
    input_: Dict[str, str],
    func_one: Callable[[str], Optional[int]],
    func_many: Callable[[Dict[str, str]], Optional[int]],
    filters: Union[str, List[str]],
):
    if isinstance(filters, str):
        if filters in input_:
            return func_one(input_[filters])
        else:
            return func_one("")
    else:
        filtered_input = {
            key: value for key, value in input_.items() if len(filters) == 0 or key in filters
        }
        return func_many(filtered_input)


def handle_file(
    raw: bytes,
    func_one: Callable[[str], Optional[int]],
    func_many: Callable[[Dict[str, str]], Optional[int]],
    filters: Union[str, List[str]],
) -> Optional[int]:
    if is_tar(raw):
        return handle_multiple(extract_tar(raw), func_one, func_many, filters)

    raw_string = raw.decode("utf-8")
    if is_ptml(raw_string):
        if len(filters) > 0:
            log("Cannot extract/filter a plain ptml file")
            return 1
        return func_one(raw_string)
    else:
        parsed_yaml = yaml_load(raw_string)
        if parsed_yaml is None:
            return 0
        return handle_multiple(parsed_yaml, func_one, func_many, filters)


@overload
def parse(raw: bytes, console: Console, filters: List[str] = []) -> None:
    ...


@overload
def parse(raw: bytes, console: None = None, filters: List[str] = []) -> str:
    ...


def parse(raw: bytes, console: Console | None = None, filters: List[str] = []) -> str | None:
    _console: Console | StringConsole = console if console else StringConsole()

    handle_file(
        raw,
        lambda x: parse_ptml_plain(x, _console),
        lambda x: parse_ptml_yaml(x, _console),
        filters,
    )

    if isinstance(_console, StringConsole):
        return _console.string
    return None
