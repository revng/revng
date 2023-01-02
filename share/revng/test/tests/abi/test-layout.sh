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

SCRIPT_DIRECTORY="$( dirname -- "$( readlink -f -- "$0"; )"; )"
TEMPORARY_DIRECTORY="$(mktemp --tmpdir tmp.abi-conversion-test.XXXXXXXXXX -d)"
trap 'rm -rf "$TEMPORARY_DIRECTORY"' EXIT

orc shell \
  llvm-dwarfdump \
  "$BINARY" \
  > "$TEMPORARY_DIRECTORY/dwarf.dump"

# Import DWARF information
revng \
    analyze \
    ImportBinary \
    "$BINARY" \
    -o="${TEMPORARY_DIRECTORY}/imported_binary.yml"

# Make sure all the primitive types are available
revng \
    analyze \
    AddPrimitiveTypes \
    "$BINARY" \
    -m="${TEMPORARY_DIRECTORY}/imported_binary.yml" \
    -o="${TEMPORARY_DIRECTORY}/reference_binary.yml"

# Convert CABIFunctionType to RawFunctionType
revng \
    analyze \
    -P="${SCRIPT_DIRECTORY}/custom-layout-pipeline.yml" \
    ConvertToRawFunctionType \
    "$BINARY" \
    --ConvertToRawFunctionType-abi="${ABI_NAME}" \
    -m="${TEMPORARY_DIRECTORY}/reference_binary.yml" \
    -o="${TEMPORARY_DIRECTORY}/downgraded_reference_binary.yml"

# Verify that conversion didn't break anything major
revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${TEMPORARY_DIRECTORY}/reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${TEMPORARY_DIRECTORY}/downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

# NOTE: this is a partial test, for the full version, see the revng-c side.
