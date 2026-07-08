#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import re
from dataclasses import dataclass
from typing import Any, List, Literal


@dataclass
class ExtraColumn:
    name: str
    label: str
    type: Literal["int", "float", "str", "bool"]  # noqa: A003
    generator: str | None

    @staticmethod
    def from_dict(input_: dict) -> "ExtraColumn":
        return ExtraColumn(input_["name"], input_["label"], input_["type"], input_.get("generator"))


@dataclass
class Download:
    name: str
    label: str

    @staticmethod
    def from_dict(input_: dict) -> "Download":
        return Download(input_["name"], input_["label"])


@dataclass
class StacktraceAggregation:
    exclude_paths: List[str]
    exclude_libs: List[str]

    @staticmethod
    def from_dict(input_: dict) -> "StacktraceAggregation":
        return StacktraceAggregation(
            input_.get("exclude_paths", []), input_.get("exclude_libs", [])
        )


@dataclass
class CrashComponentFilter:
    category: str
    type: str  # noqa: A003
    variable: str
    value: Any
    suffix: str
    label: str

    @staticmethod
    def from_dict(input_: dict) -> "CrashComponentFilter":
        return CrashComponentFilter(
            input_["category"],
            input_["type"],
            input_["variable"],
            input_["value"],
            input_["suffix"],
            input_["label"],
        )


@dataclass
class GlobalMeta:
    extra_columns: List[ExtraColumn]
    crash_components_filters: list[CrashComponentFilter]
    downloads: List[Download]
    stacktrace_aggregation: StacktraceAggregation
    flamegraph_exclude_paths: list[re.Pattern]
    notes: str | None

    @staticmethod
    def from_dict(input_: dict) -> "GlobalMeta":
        extra_columns = [ExtraColumn.from_dict(e) for e in input_.get("extra_columns", [])]
        crash_components_filters = [
            CrashComponentFilter.from_dict(e) for e in input_.get("crash_components_filters", [])
        ]
        downloads = [Download.from_dict(e) for e in input_.get("downloads", [])]
        stacktrace_aggregation = StacktraceAggregation.from_dict(
            input_.get("stacktrace_aggregation", {})
        )
        flamegraph_exclude_paths = [
            re.compile(x) for x in input_.get("flamegraph_exclude_paths", [])
        ]

        return GlobalMeta(
            extra_columns,
            crash_components_filters,
            downloads,
            stacktrace_aggregation,
            flamegraph_exclude_paths,
            input_.get("notes"),
        )
