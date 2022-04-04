#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
set -euo pipefail

DIR=$(dirname "${BASH_SOURCE[0]}")

cd "$DIR" || exit 1

python3 -m pytest "test.py::${TEST_NAME}" --binary "$BINARY_FILE" --root "$ROOT"
