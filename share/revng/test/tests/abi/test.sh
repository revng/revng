#!/bin/bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

ABI_NAME="$1"
RUNTIME_ABI_ANALYSIS_RESULT="$2"
BINARY="$3"

test -n "$ABI_NAME"
test -n "$RUNTIME_ABI_ANALYSIS_RESULT"
test -n "$BINARY"

TEMPORARY_DIR="$(mktemp --tmpdir tmp.cabi-test.XXXXXXXXXX -d)"
trap 'rm -rf "$TEMPORARY_DIR"' EXIT

# Import DWARF information
revng \
    model \
    import \
    debug-info \
    "$BINARY" \
    > "${TEMPORARY_DIR}/reference_binary.yml"

# Convert CABIFunctionType to RawFunctionType
revng \
    model \
    opt \
    --convert-all-cabi-functions-to-raw \
    --abi="${ABI_NAME}" \
    "${TEMPORARY_DIR}/reference_binary.yml" \
    > "${TEMPORARY_DIR}/downgraded_reference_binary.yml"

# Convert RawFunctionType back to CABIFunctionType
revng \
    model \
    opt \
    --convert-all-raw-functions-to-cabi \
    --abi="${ABI_NAME}" \
    "${TEMPORARY_DIR}/downgraded_reference_binary.yml" \
    > "${TEMPORARY_DIR}/upgraded_downgraded_reference_binary.yml"

# Back to RawFunctionType again
revng \
    model \
    opt \
    --convert-all-cabi-functions-to-raw \
    --abi="${ABI_NAME}" \
    "${TEMPORARY_DIR}/upgraded_downgraded_reference_binary.yml" \
    > "${TEMPORARY_DIR}/downgraded_upgraded_downgraded_reference_binary.yml"

# Check there are no differences
revng \
    ensure-rft-equivalence \
    "${TEMPORARY_DIR}/downgraded_reference_binary.yml" \
    "${TEMPORARY_DIR}/downgraded_upgraded_downgraded_reference_binary.yml"

# Verify that no step contradicts the actual state.
revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${TEMPORARY_DIR}/reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${TEMPORARY_DIR}/downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${TEMPORARY_DIR}/upgraded_downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${TEMPORARY_DIR}/downgraded_upgraded_downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"
