#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
import sys
from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, ParamSpec, TypeVar, Union

import yaml

from ...support import extract_tar, is_tar


def log(msg: str):
    sys.stderr.write(f"{msg}\n")


def yaml_load(content: str) -> Dict[str, Any]:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(content, loader)


def yaml_dump(obj: Any) -> str:
    dumper = getattr(yaml, "CDumper", yaml.Dumper)
    return yaml.dump(obj, Dumper=dumper)


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


T = TypeVar("T")
P = ParamSpec("P")


def suppress_brokenpipe(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Optional[T]:
    """When running a program with a pipe, BrokenPipeError might be raised. This signals that the
    output pipe was closed, which we expect. Suppressing the exception is not enough since it can
    also happen at shutdown, which will trigger python's unraisable hook, to remedy this we
    overwrite the default hook to ignore BrokenPipeError."""

    def new_unraisablehook(arg):
        if arg.exc_type != BrokenPipeError:
            sys.__unraisablehook__(arg)

    sys.unraisablehook = new_unraisablehook

    with suppress(BrokenPipeError):
        return func(*args, **kwargs)
    return None
