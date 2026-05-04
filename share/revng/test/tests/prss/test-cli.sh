#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail
INPUT_BINARY="$1"

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common"

start_postgres
start_rss_server

PROJECT_ID=$(python -c 'from uuid import uuid4; print(str(uuid4()))')
PROJECT_OPTS=(--project-id "$PROJECT_ID")
CMD=(revng2 project --storage-provider "rss://127.0.0.1:$RSS_SERVER_PORT/?proto=http")

# Actually run the commands
"${CMD[@]}" init "${PROJECT_OPTS[@]}" "$INPUT_BINARY"
"${CMD[@]}" artifact emit-c "${PROJECT_OPTS[@]}" --tar -o "$WORKDIR/output.tar"

# Check that the output is actually PTML
revng ptml --plain "$WORKDIR/output.tar" > /dev/null
