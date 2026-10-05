#!/usr/bin/env python3

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# This is a thin wrapper over `revng check-compatibility-with-abi` that checks
# each failure against a list of regexes.
#
# It expects the expected failure list to live in the same directory as this script,
# while mirroring the CLI of `revng check-compatibility-with-abi`.

import argparse
import os
import re
import subprocess
import sys

import yaml


class ExpectedFailure:
    """A single expected failure:
    - the ABI,
    - the test name,
    - and the reason it failed,
    each a regex.

    All three have to match for the failure to be ignored."""

    def __init__(self, abi_regex, test_regex, error_regex):
        self.abi_regex = re.compile(abi_regex)
        self.test_regex = re.compile(test_regex)
        self.error_regex = re.compile(error_regex)

    def matches(self, abi, test, error):
        return (
            bool(self.abi_regex.fullmatch(abi))
            and bool(self.test_regex.fullmatch(test))
            and bool(self.error_regex.fullmatch(error))
        )


def load_expected_failures(path):
    """Read the list of expected failures from the file pointed to by `path`."""

    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        document = yaml.safe_load(f)

    if document is None:
        return []

    if not isinstance(document, list):
        sys.exit(f"{path}: expected a list of expected failures.")

    result = []
    for entry in document:
        if not isinstance(entry, dict) or set(entry) != {"ABI", "Test", "Error"}:
            sys.exit(
                f"{path}: entry `{entry}` is not a mapping with exactly the keys "
                f"`ABI`, `Test` and `Error`."
            )

        abi_regex, test_regex, error_regex = (str(entry[key]) for key in ("ABI", "Test", "Error"))
        try:
            result.append(ExpectedFailure(abi_regex, test_regex, error_regex))
        except re.error as error:
            sys.exit(
                f"{path}: `{abi_regex}: {test_regex}: {error_regex}` is not a valid "
                f"regular expression: {error}"
            )

    return result


def verify(abi_name, model_path, runtime_artifact):
    """Run `check-compatibility-with-abi` on one model, save the `failed-tests` document
    next to it, and return the failures it reported as `(name, reason)` pairs."""

    if not os.path.exists(model_path):
        sys.exit(f"{model_path} does not exist, so it cannot be verified.")

    model_name = os.path.basename(model_path)[: -len(".yml")]
    results_path = os.path.join(os.path.dirname(model_path), model_name + ".abi-failures.yml")

    try:
        # The document goes to stderr (`dbg` is `std::cerr`), so that is what gets
        # saved.
        with open(results_path + ".stdout", "w") as standard_output, open(
            results_path, "w"
        ) as standard_error:
            subprocess.run(
                [
                    "revng",
                    "check-compatibility-with-abi",
                    f"-abi={abi_name}",
                    model_path,
                    runtime_artifact,
                ],
                stdout=standard_output,
                stderr=standard_error,
            )
    except OSError as error:
        sys.exit(f"could not run `revng check-compatibility-with-abi`: {error}")

    if os.path.getsize(results_path) == 0:
        sys.exit(
            f"`check-compatibility-with-abi` printed no results in {results_path}. "
            f"It most likely aborted before verifying anything, which is not the "
            f"same as passing."
        )

    with open(results_path, "r") as f:
        text = f.read()

    try:
        document = yaml.safe_load(text)

    except yaml.YAMLError as error:
        sys.exit(
            f"{results_path}: the output of `check-compatibility-with-abi` is not a "
            f"valid `failed-tests` document, which means it aborted instead of "
            f"reporting ({error}).\n{text}"
        )

    if not isinstance(document, dict) or "failed-tests" not in document:
        sys.exit(
            f"{results_path}: no `failed-tests` key in the output of "
            f"`check-compatibility-with-abi`.\n{text}"
        )

    failures = []
    for entry in document["failed-tests"] or []:
        # Each entry is a single-key mapping: `<test function name>: <reason>`.
        for name, reason in entry.items():
            failures.append((str(name), str(reason)))

    return model_name, failures


def main():
    parser = argparse.ArgumentParser(
        description="Verify one model against the runtime artifact and fail on any "
        "failure that is not declared as expected."
    )
    parser.add_argument(
        "-abi", required=True, metavar="ABI", help="ABI the model is expected to conform to"
    )
    parser.add_argument("model_path", help="YAML file with the model to verify")
    parser.add_argument("runtime_artifact", help="YAML file produced by running the test binary")
    parser.add_argument(
        "--expected-failures",
        help="YAML file declaring the expected failures (default: "
        "expected-abi-failures.yml next to this script)",
    )
    arguments = parser.parse_args()

    expected_failures_path = arguments.expected_failures
    if expected_failures_path is None:
        expected_failures_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "expected-abi-failures.yml"
        )

    expected_failures = load_expected_failures(expected_failures_path)

    model_name, failures = verify(arguments.abi, arguments.model_path, arguments.runtime_artifact)

    unexpected = [
        (name, reason)
        for name, reason in failures
        if not any(f.matches(arguments.abi, name, reason) for f in expected_failures)
    ]

    for name, reason in unexpected:
        print(f"({model_name}) `{name}`: {reason}", file=sys.stderr)

    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
