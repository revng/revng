#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import json
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from hashlib import shake_256
from pathlib import Path
from subprocess import run
from typing import Collection, Dict, Iterable, List

from .meta import StacktraceAggregation

SCRIPT_DIR = Path(__file__).parent.resolve()
COMPONENTS_RES = (
    re.compile(r"include/[^/]+/(?P<component>.+)/[^/]+$"),
    re.compile(r"lib/(?P<component>.+)/[^/]+$"),
)


@dataclass
class StacktraceLine:
    symbol: str
    path: str
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

    def perf_line(self, inverted: bool) -> str:
        out = [e.to_string() for e in self.lines]
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
        return StacktraceLine(outname, symbol["FileName"], symbol["Line"])
    else:
        return StacktraceLine(outname, entry["ModuleName"], None)


def stacktrace_transform(data: Iterable[str]) -> Stacktrace:
    results: List[StacktraceLine] = []
    for line in data:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            results.append(StacktraceLine("???", "???", None))
            continue

        if "Symbol" not in entry:
            results.append(StacktraceLine("???", "???", None))
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
    stacktraces: Collection[Stacktrace | None], output: Path, title: str, inverted: bool = False
):
    if len(stacktraces) == 0:
        Path(output).write_text(EMPTY_FLAMEGRAPH_SVG)
        return

    lines = ""
    for stacktrace in stacktraces:
        if stacktrace is not None:
            lines += stacktrace.perf_line(inverted)
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
) -> Dict[str, int]:
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

    return dict(counts)
