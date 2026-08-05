#!/bin/bash
set -euo pipefail

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
INPUT=$1
OUTPUT=$2

./bin/revng project dump-pipeline | \
    "$SCRIPT_DIR/generate-project-mixins.py" -o "$OUTPUT" "$INPUT"
