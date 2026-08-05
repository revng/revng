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

llvm-dwarfdump \
  "$BINARY" \
  > "$OUTPUT_DIRECTORY/dwarf.dump"

if test -e "${BINARY}.pdb"; then
    EXTRA_ARGUMENTS=(--use-pdb="${BINARY}.pdb")
fi

# Import DWARF information
revng -C "$OUTPUT_DIRECTORY" project init --overwrite --no-initial-auto-analysis "$BINARY"
revng -C "$OUTPUT_DIRECTORY" project analyze parse-binary -o /dev/null -- "${EXTRA_ARGUMENTS[@]}"

# Remove all the functions we don't find relevant, then force-override the ABI
# field of all the renaming prototypes because DWARF information is not always
# reliable
python3 \
    "${SCRIPT_DIRECTORY}/prepare-tested-model.py" \
    "$ABI_NAME" \
    "${OUTPUT_DIRECTORY}/revng.yml" \
    "${OUTPUT_DIRECTORY}/reference_binary.yml"

# Convert CABIFunctionDefinition to RawFunctionDefinition
revng pipeline run-analysis convert-functions-to-raw \
    "${OUTPUT_DIRECTORY}/reference_binary.yml" \
    -o "${OUTPUT_DIRECTORY}/downgraded_reference_binary.yml"

# Convert RawFunctionDefinition back to CABIFunctionDefinition
echo "ABI: ${ABI_NAME}" > "${OUTPUT_DIRECTORY}/convert-to-cabi-configuration.yml"
revng pipeline run-analysis convert-functions-to-cabi \
    --configuration "${OUTPUT_DIRECTORY}/convert-to-cabi-configuration.yml" \
    "${OUTPUT_DIRECTORY}/downgraded_reference_binary.yml" \
    -o "${OUTPUT_DIRECTORY}/upgraded_downgraded_reference_binary.yml"

# Back to RawFunctionDefinition again
revng pipeline run-analysis convert-functions-to-raw \
    "${OUTPUT_DIRECTORY}/upgraded_downgraded_reference_binary.yml" \
    -o "${OUTPUT_DIRECTORY}/downgraded_upgraded_downgraded_reference_binary.yml"

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
