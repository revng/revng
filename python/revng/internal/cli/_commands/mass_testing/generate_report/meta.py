#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass
from typing import List, Literal


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
class GlobalMeta:
    extra_columns: List[ExtraColumn]
    downloads: List[Download]
    stacktrace_aggregation: StacktraceAggregation
    notes: str | None

    @staticmethod
    def from_dict(input_: dict) -> "GlobalMeta":
        extra_columns = [ExtraColumn.from_dict(e) for e in input_.get("extra_columns", [])]
        downloads = [Download.from_dict(e) for e in input_.get("downloads", [])]
        stacktrace_aggregation = StacktraceAggregation.from_dict(
            input_.get("stacktrace_aggregation", {})
        )
        return GlobalMeta(extra_columns, downloads, stacktrace_aggregation, input_.get("notes"))
