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
mkdir -p "${OUTPUT_DIRECTORY}/project"
revng -C "${OUTPUT_DIRECTORY}/project" project init --overwrite --no-initial-auto-analysis "$BINARY"
revng -C "${OUTPUT_DIRECTORY}/project" project analyze parse-binary -o /dev/null -- "${EXTRA_ARGUMENTS[@]}"

# Remove all the functions we don't find relevant, then force-override the ABI
# field of all the renaming prototypes because DWARF information is not always
# reliable
mkdir -p "${OUTPUT_DIRECTORY}/models"
python3 \
    "${SCRIPT_DIRECTORY}/prepare-tested-model.py" \
    "$ABI_NAME" \
    "${OUTPUT_DIRECTORY}/project/revng.yml" \
    "${OUTPUT_DIRECTORY}/models/1-reference-binary.yml"

# Convert CABIFunctionDefinition to RawFunctionDefinition
revng pipeline run-analysis convert-functions-to-raw \
    "${OUTPUT_DIRECTORY}/models/1-reference-binary.yml" \
    -o "${OUTPUT_DIRECTORY}/models/2-downgraded-reference-binary.yml"

# Convert RawFunctionDefinition back to CABIFunctionDefinition
#
# The conversion is restricted to a list of "safe" ABIs by default, and the ABIs
# tested here are not all part of it, so request the "unsafe" mode explicitly.
{
  echo "ABI: ${ABI_NAME}"
  echo "Mode: unsafe"
} > "${OUTPUT_DIRECTORY}/convert-to-cabi-configuration.yml"
revng pipeline run-analysis convert-functions-to-cabi \
    --configuration "${OUTPUT_DIRECTORY}/convert-to-cabi-configuration.yml" \
    "${OUTPUT_DIRECTORY}/models/2-downgraded-reference-binary.yml" \
    -o "${OUTPUT_DIRECTORY}/models/3-upgraded-downgraded-reference-binary.yml"

# Back to RawFunctionDefinition again
revng pipeline run-analysis convert-functions-to-raw \
    "${OUTPUT_DIRECTORY}/models/3-upgraded-downgraded-reference-binary.yml" \
    -o "${OUTPUT_DIRECTORY}/models/4-downgraded-upgraded-downgraded-reference-binary.yml"

# Verify that no step contradicts the actual state.
EXPECTED_FAILURES="${SCRIPT_DIRECTORY}/expected-failures"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    --expected-failures="${EXPECTED_FAILURES}/floating-point.yml" \
    --expected-failures="${EXPECTED_FAILURES}/arm-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/microsoft-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/mips-specific.yml" \
    --verify-type=CABIFunctionDefinition \
    "${OUTPUT_DIRECTORY}/models/1-reference-binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    --expected-failures="${EXPECTED_FAILURES}/floating-point.yml" \
    --expected-failures="${EXPECTED_FAILURES}/arm-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/microsoft-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/mips-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/rft-insufficiency.yml" \
    --expected-failures="${EXPECTED_FAILURES}/apple-specific.yml" \
    --verify-type=RawFunctionDefinition \
    "${OUTPUT_DIRECTORY}/models/2-downgraded-reference-binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    --expected-failures="${EXPECTED_FAILURES}/floating-point.yml" \
    --expected-failures="${EXPECTED_FAILURES}/arm-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/microsoft-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/mips-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/rft-insufficiency.yml" \
    --expected-failures="${EXPECTED_FAILURES}/non-cft.yml" \
    --verify-type=CABIFunctionDefinition \
    "${OUTPUT_DIRECTORY}/models/3-upgraded-downgraded-reference-binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    --expected-failures="${EXPECTED_FAILURES}/floating-point.yml" \
    --expected-failures="${EXPECTED_FAILURES}/arm-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/microsoft-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/mips-specific.yml" \
    --expected-failures="${EXPECTED_FAILURES}/rft-insufficiency.yml" \
    --expected-failures="${EXPECTED_FAILURES}/apple-specific.yml" \
    --verify-type=RawFunctionDefinition \
    "${OUTPUT_DIRECTORY}/models/4-downgraded-upgraded-downgraded-reference-binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}"

# Check there are no differences
revng \
    ensure-rft-equivalence \
    "${OUTPUT_DIRECTORY}/models/2-downgraded-reference-binary.yml" \
    "${OUTPUT_DIRECTORY}/models/4-downgraded-upgraded-downgraded-reference-binary.yml"
