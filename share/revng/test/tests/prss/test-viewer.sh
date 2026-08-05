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
start_rss_viewer

PROJECT_ID=$(python -c 'from uuid import uuid4; print(str(uuid4()))')
PROJECT_OPTS=(--project-id "$PROJECT_ID")
CMD=(revng project --storage-provider "rss://127.0.0.1:$RSS_SERVER_PORT/?proto=http")

# Actually run the commands
"${CMD[@]}" init "${PROJECT_OPTS[@]}" "$INPUT_BINARY"
"${CMD[@]}" artifact emit-c "${PROJECT_OPTS[@]}" --tar -o /dev/null

# Check that the viewer has the data
PDV_BASE_URL="http://127.0.0.1:$RSS_VIEWER_PORT"
CURL_CMD=(curl -s --fail -H "x-project-id: $PROJECT_ID")

# Check epoch
"${CURL_CMD[@]}" "$PDV_BASE_URL/epoch" -o "$WORKDIR/epoch.json"
jq \
    'if .epoch | type != "number" then error("Epoch is not a number") else . end' \
    "$WORKDIR/epoch.json" > /dev/null
EPOCH=$(jq -r .epoch "$WORKDIR/epoch.json")

# Check model
"${CURL_CMD[@]}" "$PDV_BASE_URL/model" | \
    jq -r .model | \
    revng model opt -verify > /dev/null

# Check pipeline-description
"${CURL_CMD[@]}" "$PDV_BASE_URL/pipeline-description" -o "$WORKDIR/pipeline-description.yml"
yq . "$WORKDIR/pipeline-description.yml" > /dev/null

# Get the savepoint_id and container_id of `emit-c`
SAVEPOINT_ID=$(yq -r '.artifacts[] | select(.name == "emit-c") | .savepoint_id' "$WORKDIR/pipeline-description.yml")
CONTAINER_ID=$(yq -r '.artifacts[] | select(.name == "emit-c") | .container' "$WORKDIR/pipeline-description.yml")

"${CURL_CMD[@]}" \
    "$PDV_BASE_URL/list-objects?savepoint_id=${SAVEPOINT_ID}&container_id=${CONTAINER_ID}" \
    -o "$WORKDIR/emit-c-objects.json"
jq \
    'if . | type != "array" then error("Objects is not an array") else . end' \
    "$WORKDIR/emit-c-objects.json" \
    > /dev/null

# Get the first object ID to run object tests
FIRST_OBJECT=$(jq -r '.[0]' "$WORKDIR/emit-c-objects.json")
FIRST_OBJECT_ESCAPED=$(sed -e 's;/;%2F;g' -e 's;:;%3A;g' <<< "$FIRST_OBJECT")

# Retrieve the first object
QUERY="savepoint_id=${SAVEPOINT_ID}&container_id=${CONTAINER_ID}&object_id=${FIRST_OBJECT_ESCAPED}"
"${CURL_CMD[@]}" "$PDV_BASE_URL/object?${QUERY}" | \
    zstdcat | revng ptml > /dev/null

# Retrieve the first object but use `decompress`
QUERY2="${QUERY}&decompress=zstd"
"${CURL_CMD[@]}" "$PDV_BASE_URL/object?${QUERY2}" | revng ptml > /dev/null

# Retrieve the first object, use `decompress` and `Content-Encoding`
"${CURL_CMD[@]}" -H 'Content-Encoding: zstd' "$PDV_BASE_URL/object?${QUERY2}" | \
    zstdcat | revng ptml > /dev/null

# Test that with the wrong epoch we get an error
QUERY3="${QUERY}&epoch=$((EPOCH-1))"
HTTP_CODE=$("${CURL_CMD[@]}" "$PDV_BASE_URL/object?${QUERY3}" \
                -o /dev/null --write-out '%{http_code}' || true)
[[ "$HTTP_CODE" -eq 409 ]]
