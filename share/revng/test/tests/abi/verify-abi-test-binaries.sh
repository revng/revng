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
    import-binary \
    "$BINARY" \
    --use-pdb="${BINARY}.pdb" \
    -o="${TEMPORARY_DIRECTORY}/imported_binary.yml"

# Remove all the functions we don't find relevant, then force-override the ABI
# field of all the renaming prototypes because DWARF information is not always
# reliable
python3 \
    "${SCRIPT_DIRECTORY}/prepare-tested-model.py" \
    "$ABI_NAME" \
    "${TEMPORARY_DIRECTORY}/imported_binary.yml" \
    "${TEMPORARY_DIRECTORY}/prepared_binary.yml"

# Make sure all the primitive types are available
revng \
    analyze \
    add-primitive-types \
    "$BINARY" \
    -m="${TEMPORARY_DIRECTORY}/prepared_binary.yml" \
    -o="${TEMPORARY_DIRECTORY}/reference_binary.yml"

# Convert CABIFunctionType to RawFunctionType
revng \
    analyze \
    -P="${SCRIPT_DIRECTORY}/custom-conversion-pipeline.yml" \
    convert-functions-to-raw \
    "$BINARY" \
    -m="${TEMPORARY_DIRECTORY}/reference_binary.yml" \
    -o="${TEMPORARY_DIRECTORY}/downgraded_reference_binary.yml"

# Convert RawFunctionType back to CABIFunctionType
revng \
    analyze \
    -P="${SCRIPT_DIRECTORY}/custom-conversion-pipeline.yml" \
    convert-functions-to-cabi \
    "$BINARY" \
    --convert-functions-to-cabi-abi="${ABI_NAME}" \
    -m="${TEMPORARY_DIRECTORY}/downgraded_reference_binary.yml" \
    -o="${TEMPORARY_DIRECTORY}/upgraded_downgraded_reference_binary.yml"

# Back to RawFunctionType again
revng \
    analyze \
    -P="${SCRIPT_DIRECTORY}/custom-conversion-pipeline.yml" \
    convert-functions-to-raw \
    "$BINARY" \
    -m="${TEMPORARY_DIRECTORY}/upgraded_downgraded_reference_binary.yml" \
    -o="${TEMPORARY_DIRECTORY}/downgraded_upgraded_downgraded_reference_binary.yml"

# Verify that no step contradicts the actual state.
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

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${TEMPORARY_DIRECTORY}/upgraded_downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${TEMPORARY_DIRECTORY}/downgraded_upgraded_downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

# Check there are no differences
revng \
    ensure-rft-equivalence \
    "${TEMPORARY_DIRECTORY}/downgraded_reference_binary.yml" \
    "${TEMPORARY_DIRECTORY}/downgraded_upgraded_downgraded_reference_binary.yml"
