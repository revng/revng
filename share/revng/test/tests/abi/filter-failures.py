#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Filter the failures reported by `check-compatibility-with-abi` against a
# list of known and expected failures.
#
# The tool prints a yaml document containing a single `failed-tests` map.
#
# The value is either a mapping of test names to failure reasons, or a
# sequence of single-entry mappings (which is what the textual
# `- test: reason` report turns into once parsed).
#
# An empty map means a passing test. Non-empty map might still mean a success,
# hence why filtering (this script) is required.
#
# This script accepts a set of expected failures and verifies the input
# document against them. Both directions are checked:
#
#   (a) a failure that no entry accounts for is reported (a real regression);
#   (b) an entry whose (ABI, test) pair matched nothing is also reported.
#
# Each file is a top-level sequence of entries:
#
#   - ABIs: [a-list-of-abis]
#     tests: [a-list-of-tests]  # noqa: E800
#     errors: [a-list-of-error-messages]  # noqa: E800
#     can-succeed: false
#
# See the README.md in the `expected-failures` directory for some information
# on why those specific failures are marked as expected.

import argparse
import os
import sys
from dataclasses import dataclass, field

import yaml


@dataclass
class Failure:
    test: str
    reason: str


@dataclass
class Expectation:
    abis: list[str]
    tests: list[str]
    errors: list[str] = field(default_factory=list)
    can_succeed: bool = False


def failure_entries(entries: object) -> list[tuple[str, str]]:
    if entries is None:
        return []

    if isinstance(entries, dict):
        pairs = list(entries.items())
    elif isinstance(entries, list):
        pairs = []
        for entry in entries:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(f"malformed `failed-tests` entry: {entry!r}")
            pairs.extend(entry.items())
    else:
        raise ValueError("`failed-tests` is neither a map nor a sequence of maps")

    for test, reason in pairs:
        if not isinstance(test, str) or not isinstance(reason, str):
            raise ValueError(f"malformed `failed-tests` entry: {test!r}: {reason!r}")

    return pairs


def parse_failures(text: str) -> list[Failure]:
    # A truncated or otherwise broken report must not pass for a clean run.
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError("expected a YAML document containing a `failed-tests` map")
    if "failed-tests" not in document:
        raise ValueError("no `failed-tests` map found in the input")

    return [Failure(test, reason) for test, reason in failure_entries(document["failed-tests"])]


def parse_expectation(index: int, path: str, entry: object) -> Expectation:
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: entry {index} is not a mapping")

    unknown = set(entry) - {"ABIs", "tests", "errors", "can-succeed"}
    if unknown:
        raise ValueError(f"{path}: entry {index} has unknown keys: {sorted(unknown)}")

    def string_list(key: str, required: bool) -> list[str]:
        if key not in entry:
            if required:
                raise ValueError(f"{path}: entry {index} is missing `{key}`")
            return []
        value = entry[key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{path}: entry {index} has a malformed `{key}`")
        return value

    abis = string_list("ABIs", required=True)
    tests = string_list("tests", required=True)
    errors = string_list("errors", required=False)
    can_succeed = entry.get("can-succeed", False)

    if not abis:
        raise ValueError(f"{path}: entry {index} has an empty `ABIs`")
    if not tests:
        raise ValueError(f"{path}: entry {index} has an empty `tests`")
    if not isinstance(can_succeed, bool):
        raise ValueError(f"{path}: entry {index} has a malformed `can-succeed`")

    return Expectation(abis, tests, errors, can_succeed)


def load_expectations(paths: list[str], abi_name: str) -> list[Expectation]:
    expectations = []

    for path in paths:
        with open(path) as stream:
            document = yaml.safe_load(stream)

        if document is None:
            continue
        if not isinstance(document, list):
            raise ValueError(f"{path}: expected a top-level sequence of entries")

        for index, entry in enumerate(document):
            expectation = parse_expectation(index, path, entry)
            if abi_name in expectation.abis:
                expectations.append(expectation)

    return expectations


def filter_failures(failures: list[Failure], expectations: list[Expectation]) -> list[Failure]:
    remaining = []
    matched: list[set[str]] = [set() for _ in expectations]

    for failure in failures:
        was_expected = False
        for index, expectation in enumerate(expectations):
            if failure.test not in expectation.tests:
                continue
            if expectation.errors and failure.reason not in expectation.errors:
                continue
            matched[index].add(failure.test)
            was_expected = True

        if not was_expected:
            remaining.append(failure)

    for index, expectation in enumerate(expectations):
        if expectation.can_succeed:
            # Explicitly allowed to describe tests that may succeed, so the
            # stale-expectation check does not apply to it.
            continue

        if len(matched[index]) == len(expectation.tests):
            # All the expected failures were matched.
            continue

        for test in expectation.tests:
            if test not in matched[index]:
                remaining.append(
                    Failure(
                        test,
                        f"An entry in the expectation {index} didn't match a failure.",
                    )
                )

    return remaining


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter the failures reported by check-compatibility-with-abi."
    )
    parser.add_argument("abi", help="the ABI under test")
    parser.add_argument("input", help="the failure report to filter")
    parser.add_argument(
        "expected_failures",
        nargs="*",
        metavar="EXPECTED_FAILURES",
        help="a list of known and expected failures",
    )
    args = parser.parse_args()

    try:
        with open(args.input) as stream:
            failures = parse_failures(stream.read())
        expectations = load_expectations(args.expected_failures, args.abi)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"filter-failures: {error}", file=sys.stderr)
        return 2

    remaining = filter_failures(failures, expectations)
    if remaining:
        print(f"{os.path.basename(args.input)}:")
        for failure in remaining:
            print(f"- {failure.test}: {failure.reason}")

    return len(remaining)


if __name__ == "__main__":
    sys.exit(main())
