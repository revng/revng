#!/bin/bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

ABI_NAME="$1"
RUNTIME_ABI_ANALYSIS_RESULT="$2"
BINARY="$3"
OUTPUT_DIRECTORY="$4"

test -n "$ABI_NAME"
test -n "$RUNTIME_ABI_ANALYSIS_RESULT"
test -n "$BINARY"
test -n "$OUTPUT_DIRECTORY"

SCRIPT_DIRECTORY="$( dirname -- "$( readlink -f -- "$0"; )"; )"

orc shell \
  llvm-dwarfdump \
  "$BINARY" \
  > "$OUTPUT_DIRECTORY/dwarf.dump"

# Import DWARF information
revng \
    analyze \
    import-binary \
    "$BINARY" \
    --use-pdb="${BINARY}.pdb" \
    -o="${OUTPUT_DIRECTORY}/imported_binary.yml"

# Remove all the functions we don't find relevant, then force-override the ABI
# field of all the renaming prototypes because DWARF information is not always
# reliable
python3 \
    "${SCRIPT_DIRECTORY}/prepare-tested-model.py" \
    "$ABI_NAME" \
    "${OUTPUT_DIRECTORY}/imported_binary.yml" \
    "${OUTPUT_DIRECTORY}/prepared_binary.yml"

# Make sure all the primitive types are available
revng \
    analyze \
    add-primitive-types \
    "$BINARY" \
    -m="${OUTPUT_DIRECTORY}/prepared_binary.yml" \
    -o="${OUTPUT_DIRECTORY}/reference_binary.yml"

# Convert CABIFunctionDefinition to RawFunctionDefinition
revng \
    analyze \
    -P="${SCRIPT_DIRECTORY}/custom-conversion-pipeline.yml" \
    convert-functions-to-raw \
    "$BINARY" \
    -m="${OUTPUT_DIRECTORY}/reference_binary.yml" \
    -o="${OUTPUT_DIRECTORY}/downgraded_reference_binary.yml"

# Convert RawFunctionDefinition back to CABIFunctionDefinition
revng \
    analyze \
    -P="${SCRIPT_DIRECTORY}/custom-conversion-pipeline.yml" \
    convert-functions-to-cabi \
    "$BINARY" \
    --convert-functions-to-cabi-abi="${ABI_NAME}" \
    -m="${OUTPUT_DIRECTORY}/downgraded_reference_binary.yml" \
    -o="${OUTPUT_DIRECTORY}/upgraded_downgraded_reference_binary.yml"

# Back to RawFunctionDefinition again
revng \
    analyze \
    -P="${SCRIPT_DIRECTORY}/custom-conversion-pipeline.yml" \
    convert-functions-to-raw \
    "$BINARY" \
    -m="${OUTPUT_DIRECTORY}/upgraded_downgraded_reference_binary.yml" \
    -o="${OUTPUT_DIRECTORY}/downgraded_upgraded_downgraded_reference_binary.yml"

# Verify that no step contradicts the actual state.
revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${OUTPUT_DIRECTORY}/reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${OUTPUT_DIRECTORY}/downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${OUTPUT_DIRECTORY}/upgraded_downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    "${OUTPUT_DIRECTORY}/downgraded_upgraded_downgraded_reference_binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

# Check there are no differences
revng \
    ensure-rft-equivalence \
    "${OUTPUT_DIRECTORY}/downgraded_reference_binary.yml" \
    "${OUTPUT_DIRECTORY}/downgraded_upgraded_downgraded_reference_binary.yml"
