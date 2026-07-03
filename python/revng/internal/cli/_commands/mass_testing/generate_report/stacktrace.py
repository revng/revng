#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from hashlib import shake_256
from pathlib import Path
from subprocess import run
from typing import TYPE_CHECKING, Any, Collection, Dict, Iterable, List, Protocol

from .meta import StacktraceAggregation

if TYPE_CHECKING:
    from .test_directory import TestDirectory

SCRIPT_DIR = Path(__file__).parent.resolve()
COMPONENTS_RES = (
    re.compile(r"include/[^/]+/(?P<component>.+)/[^/]+$"),
    re.compile(r"lib/(?P<component>.+)/[^/]+$"),
)


def _percentile(data: Iterable[float], p: float):
    """
    This calculates the n-th percentile (even fractional) given a set of data
    points. This uses the linear interpolation between closest ranks method to
    compute the percentile, which works better than `statistics.quantiles`.
    """

    sorted_data = sorted(data)
    n = len(sorted_data)

    assert len(sorted_data) > 0
    assert 0.0 <= p <= 100.0

    pos = p * n / 100.0 - 0.5
    if pos <= 0:
        return float(sorted_data[0])
    if pos >= n - 1:
        return float(sorted_data[-1])

    low = math.floor(pos)
    fraction = pos - low
    return sorted_data[low] + fraction * (sorted_data[low + 1] - sorted_data[low])


@dataclass
class StacktraceLine:
    symbol: str
    path: str
    module: str
    line: str | None

    def __post_init__(self):
        self.path = os.path.normpath(self.path)

    @cached_property
    def normalized_path(self):
        path = self.path
        for prefix in ("include", "lib", "lib64", "libexec"):
            full_prefix = f"/{prefix}/"
            index = path.find(full_prefix)
            if index == 0:
                return f"/{prefix}/{path.rsplit(full_prefix, 1)[1]}"
            elif index > 0:
                return f"{prefix}/{path.rsplit(full_prefix, 1)[1]}"
            if path.startswith(f"{prefix}/"):
                return path

        if path.count("/") < 4:
            return path
        else:
            return "/".join(path.rsplit("/", 4)[1:])

    def to_string(self) -> str:
        res = f"{self.symbol} at {self.path}"
        if self.line is not None:
            res += f":{self.line}"
        return res.replace(";", "")


class Stacktrace(Sequence):
    def __init__(self, lines: Iterable[StacktraceLine]):
        self.lines = list(lines)

    @cached_property
    def id_(self):
        hash_ = shake_256()
        for line in self.lines:
            hash_.update(line.to_string().encode("utf-8"))
            hash_.update(b"\0")
        return hash_.hexdigest(4)

    def _perf_lines(self, exclude_paths: list[re.Pattern], max_length: int | None) -> list[str]:
        out: list[str] = []
        excluded_paths = 0
        for index, element in enumerate(self.lines):
            if max_length is not None and len(out) >= max_length:
                out.append(f"... ({len(self.lines) - index} frame(s) skipped)")
                break

            if any(p.search(element.module) for p in exclude_paths):
                excluded_paths += 1
                continue

            if excluded_paths > 0:
                out.append(f"... ({excluded_paths} frame(s) excluded)")
                excluded_paths = 0

            out.append(element.to_string())

        return out

    def effective_length(self, exclude_paths: list[re.Pattern]) -> int:
        return len(self._perf_lines(exclude_paths, None))

    def perf_line(self, inverted: bool, max_length: int, exclude_paths: list[re.Pattern]) -> str:
        out = self._perf_lines(exclude_paths, max_length)
        if not inverted:
            out.append(self.id_)
        else:
            out.insert(0, self.id_)
        return ";".join(out if not inverted else reversed(out)) + " 1\n"

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def stacktrace_transform_entry(entry: dict, symbol: dict) -> StacktraceLine:
    if symbol["FunctionName"]:
        outname = symbol["FunctionName"]
        if symbol["StartAddress"]:
            start_addr = int(symbol["StartAddress"], 16)
            addr = int(entry["Address"], 16)
            outname += f"+{addr - start_addr:#x}"
    else:
        outname = entry["Address"]

    if symbol["FileName"]:
        return StacktraceLine(outname, symbol["FileName"], entry["ModuleName"], symbol["Line"])
    else:
        return StacktraceLine(outname, entry["ModuleName"], entry["ModuleName"], None)


def stacktrace_transform(data: Iterable[str]) -> Stacktrace:
    results: List[StacktraceLine] = []
    for line in data:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            results.append(StacktraceLine("???", "???", "???", None))
            continue

        if "Symbol" not in entry:
            results.append(StacktraceLine("???", "???", "???", None))
            continue
        for symbol in entry["Symbol"]:
            results.append(stacktrace_transform_entry(entry, symbol))

    return Stacktrace(reversed(results))


EMPTY_FLAMEGRAPH_SVG = """<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version="1.1" width="1920" height="60"
     viewBox="0 0 1920 60" xmlns="http://www.w3.org/2000/svg">
  <text  x="960.00" y="24" >This flamegraph is empty</text>
</svg>
"""


def generate_flamegraph(
    stacktraces: Collection[Stacktrace | None],
    output: Path,
    title: str,
    exclude_paths: list[re.Pattern],
    inverted: bool = False,
):
    if len(stacktraces) == 0:
        Path(output).write_text(EMPTY_FLAMEGRAPH_SVG)
        return

    lengths = [st.effective_length(exclude_paths) if st is not None else 1 for st in stacktraces]
    max_length = int(_percentile(lengths, 95))

    lines = ""
    for stacktrace in stacktraces:
        if stacktrace is not None:
            lines += stacktrace.perf_line(inverted, max_length, exclude_paths)
        else:
            lines += "no stack trace 1\n"

    opts = [
        "--color=chain",
        "--width=1920",
        "--minwidth=0",
        "--countname=run(s)",
        f"--title={title}",
    ]
    if inverted:
        opts.append("--inverted")

    with open(output, "wb") as f:
        run(("flamegraph.pl", *opts), input=lines.encode("utf-8"), stdout=f, check=True)


def find_component(stacktrace: Stacktrace, aggregation_rules: StacktraceAggregation):
    good_lines = []
    for line in stacktrace:
        if any(e in line.path for e in aggregation_rules.exclude_paths) or not any(
            line.path.endswith(e) for e in (".h", ".cpp")
        ):
            continue
        good_lines.append(line)

    for line in good_lines:
        for regex in COMPONENTS_RES:
            if match := regex.match(line.normalized_path):
                component = match.groupdict()["component"]
                if all(not re.search(f"^{e}$", component) for e in aggregation_rules.exclude_libs):
                    return component
    return None


def generate_crash_components(
    stacktraces: Collection[Stacktrace | None], aggregation_rules: StacktraceAggregation
) -> list[tuple[str, int]]:
    counts: Dict[str, int] = defaultdict(lambda: 0)
    for stacktrace in stacktraces:
        if stacktrace is None:
            counts["Other"] += 1
            continue

        component = find_component(stacktrace, aggregation_rules)
        if component is None:
            counts["Other"] += 1
        else:
            counts[component] += 1

    return list(counts.items())


class StacktraceFilter(Protocol):
    def __init__(self, variable: str, value: Any): ...

    def filter_(self, tests: list[TestDirectory]) -> list[Stacktrace | None]: ...

    def suffix(self) -> str: ...


class _PercentileFilter:
    def __init__(self, variable: str, value: int):
        self.variable = variable
        self.value = value

    def filter_(self, tests: list[TestDirectory]) -> list[Stacktrace | None]:
        if len(tests) == 0:
            return []

        values = [float(t.get_meta(self.variable)) for t in tests]
        limit = _percentile(values, self.value)
        return [t.stacktrace for index, t in enumerate(tests) if values[index] < limit]

    def suffix(self) -> str:
        return f"{self.value}th_percentile_on_{self.variable}"


STACKTRACE_FILTERS: dict[str, type[StacktraceFilter]] = {"percentile": _PercentileFilter}


def get_filter(type_: str, variable: str, value: Any) -> StacktraceFilter:
    return STACKTRACE_FILTERS[type_](variable, value)
