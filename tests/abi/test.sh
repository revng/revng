#!/bin/bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -eu
set -o pipefail

ABI_NAME="$1"
RUNTIME_ABI_ANALYSIS_RESULT="$2"
BINARY="$3"

test -n "$ABI_NAME"
test -n "$RUNTIME_ABI_ANALYSIS_RESULT"
test -n "$BINARY"

mkdir -p "model"

# Import DWARF information
revng \
    model \
    import \
    dwarf \
    "$BINARY" \
    > "model/reference_binary.yml"

# Convert CABIFunctionType to RawFunctionType
revng \
    model \
    opt \
    --convert-all-cabi-functions-to-raw \
    --abi="${ABI_NAME}" \
    "model/reference_binary.yml" \
    > "model/downgraded_reference_binary.yml"

# Convert RawFunctionType back to CABIFunctionType
revng \
    model \
    opt \
    --convert-all-raw-functions-to-cabi \
    --abi="${ABI_NAME}" \
    "model/downgraded_reference_binary.yml" \
    > "model/upgraded_downgraded_reference_binary.yml"

# Back to RawFunctionType again
revng \
    model \
    opt \
    --convert-all-cabi-functions-to-raw \
    --abi="${ABI_NAME}" \
    "model/upgraded_downgraded_reference_binary.yml" \
    > "model/downgraded_upgraded_downgraded_reference_binary.yml"

# Check there are no differences
revng \
    model \
    diff \
    "model/downgraded_reference_binary.yml" \
    "model/downgraded_upgraded_downgraded_reference_binary.yml"

# Verify
revng \
    abi-verify \
    -abi="${ABI_NAME}" \
    "model/downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"
