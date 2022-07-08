#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
import sys
from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from xml.dom import Node
from xml.dom.minidom import Document, parseString

import yaml


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


T = TypeVar("T")


def handle_file(
    raw: str,
    func_one: Callable[[str], T],
    func_many: Callable[[Dict[str, str]], T],
    filters: Union[str, List[str]],
) -> T:
    if is_ptml(raw):
        if len(filters) > 0:
            log("Cannot extract/filter a plain ptml file")
            sys.exit(1)

        return func_one(raw)
    else:
        parsed_yaml = yaml_load(raw)
        if isinstance(filters, str):
            if filters in parsed_yaml:
                return func_one(parsed_yaml[filters])
            else:
                return func_one("")
        else:
            filtered_yaml = {
                key: value
                for key, value in parsed_yaml.items()
                if len(filters) == 0 or key in filters
            }
            return func_many(filtered_yaml)


def strip_ptml(content: str) -> str:
    dom = parseString(content)
    result: List[str] = []
    _strip_ptml(dom, result)
    return "".join(result)


def _strip_ptml(node: Document, parts: List[str]):
    for node in node.childNodes:
        if node.nodeType == Node.TEXT_NODE:
            parts.append(node.nodeValue)
        elif node.nodeType == Node.ELEMENT_NODE:
            _strip_ptml(node, parts)
        else:
            pass


def suppress_brokenpipe(func: Callable[..., T], *args) -> T:
    """When running a program with a pipe, BrokenPipeError might be raised. This signals that the
    output pipe was closed, which we expect. Suppressing the exception is not enough since it can
    also happen at shutdown, which will trigger python's unraisable hook, to remedy this we
    overwrite the default hook to ignore BrokenPipeError."""

    def new_unraisablehook(arg):
        if arg.exc_type != BrokenPipeError:
            sys.__unraisablehook__(arg)

    sys.unraisablehook = new_unraisablehook

    with suppress(BrokenPipeError):
        result = func(*args)
    return result
