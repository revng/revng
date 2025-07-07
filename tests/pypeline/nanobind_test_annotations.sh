#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
ROOT_DIR="$1"
shift

"$ROOT_DIR/scripts/nanobind_generate_stubs.py" \
    -m "revng.internal._cpp_pypeline" \
    "$@" | "$SCRIPT_DIR/nanobind_test_annotations.py"
