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

# Look for any problems with the models produced above.
#
# Note that `check-compatibility-with-abi` will return *0* even if some
# functions *fail!* Non-zero exit code is only possible on serious issues (like
# missing/broken inputs).
#
# The goal of this stage is to list *all* the failures to be analyzed separately
# (having full lists simplifies debugging by a lot).
#
# See `filter-failures.py` invocations below for the actual checks.
mkdir -p "${OUTPUT_DIRECTORY}/unfiltered"
mkdir -p "${OUTPUT_DIRECTORY}/layouts"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    --save-layouts="${OUTPUT_DIRECTORY}/layouts/1-reference-binary.yml" \
    CABIFunctionDefinition \
    "${OUTPUT_DIRECTORY}/models/1-reference-binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}" \
    > "${OUTPUT_DIRECTORY}/unfiltered/1-reference-binary.yml"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    --save-layouts="${OUTPUT_DIRECTORY}/layouts/2-downgraded-reference-binary.yml" \
    RawFunctionDefinition \
    "${OUTPUT_DIRECTORY}/models/2-downgraded-reference-binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}" \
    > "${OUTPUT_DIRECTORY}/unfiltered/2-downgraded-reference-binary.yml"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    --save-layouts="${OUTPUT_DIRECTORY}/layouts/3-upgraded-downgraded-reference-binary.yml" \
    CABIFunctionDefinition \
    "${OUTPUT_DIRECTORY}/models/3-upgraded-downgraded-reference-binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}" \
    > "${OUTPUT_DIRECTORY}/unfiltered/3-upgraded-downgraded-reference-binary.yml"

revng \
    check-compatibility-with-abi \
    -abi="${ABI_NAME}" \
    --save-layouts="${OUTPUT_DIRECTORY}/layouts/4-downgraded-upgraded-downgraded-reference-binary.yml" \
    RawFunctionDefinition \
    "${OUTPUT_DIRECTORY}/models/4-downgraded-upgraded-downgraded-reference-binary.yml" \
    "${RUNTIME_ABI_ANALYSIS_RESULT}" \
    > "${OUTPUT_DIRECTORY}/unfiltered/4-downgraded-upgraded-downgraded-reference-binary.yml"

# All the failure lists are recorded, now check whether tests should pass or not
# by talking the expected failure lists into account.
#
# For the explanation on the specific lists, see the comments inside the
# corresponding files.
EXPECTED_FAILURES="${SCRIPT_DIRECTORY}/expected-failures"

# Notice that different stages use different sets of expected failures.
#
# To improve the readability of this script, let's group the entries based on
# which stages they apply to.
UNIVERSAL_FAILURES=(
    "${EXPECTED_FAILURES}/floating-point.yml"
    "${EXPECTED_FAILURES}/arm.yml"
    "${EXPECTED_FAILURES}/microsoft.yml"
    "${EXPECTED_FAILURES}/mips.yml"
)
RFT_STAGE_FAILURES=(
    "${EXPECTED_FAILURES}/RFTs.yml"
    "${EXPECTED_FAILURES}/apple.yml"
)
CFT_STAGE_FAILURES=(
    "${EXPECTED_FAILURES}/RFTs.yml"
    "${EXPECTED_FAILURES}/non-CFT.yml"
)

mkdir -p "${OUTPUT_DIRECTORY}/filtered"

python3 "${SCRIPT_DIRECTORY}/filter-failures.py" \
    "${ABI_NAME}" \
    "${OUTPUT_DIRECTORY}/unfiltered/1-reference-binary.yml" \
    "${UNIVERSAL_FAILURES[@]}" \
  | tee "${OUTPUT_DIRECTORY}/filtered/1-reference-binary.yml"

python3 "${SCRIPT_DIRECTORY}/filter-failures.py" \
    "${ABI_NAME}" \
    "${OUTPUT_DIRECTORY}/unfiltered/2-downgraded-reference-binary.yml" \
    "${UNIVERSAL_FAILURES[@]}" "${RFT_STAGE_FAILURES[@]}" \
  | tee "${OUTPUT_DIRECTORY}/filtered/2-downgraded-reference-binary.yml"

python3 "${SCRIPT_DIRECTORY}/filter-failures.py" \
    "${ABI_NAME}" \
    "${OUTPUT_DIRECTORY}/unfiltered/3-upgraded-downgraded-reference-binary.yml" \
    "${UNIVERSAL_FAILURES[@]}" "${CFT_STAGE_FAILURES[@]}" \
  | tee "${OUTPUT_DIRECTORY}/filtered/3-upgraded-downgraded-reference-binary.yml"

python3 "${SCRIPT_DIRECTORY}/filter-failures.py" \
    "${ABI_NAME}" \
    "${OUTPUT_DIRECTORY}/unfiltered/4-downgraded-upgraded-downgraded-reference-binary.yml" \
    "${UNIVERSAL_FAILURES[@]}" "${RFT_STAGE_FAILURES[@]}" \
  | tee "${OUTPUT_DIRECTORY}/filtered/4-downgraded-upgraded-downgraded-reference-binary.yml"

# Check there are no differences
revng \
    ensure-rft-equivalence \
    "${OUTPUT_DIRECTORY}/models/2-downgraded-reference-binary.yml" \
    "${OUTPUT_DIRECTORY}/models/4-downgraded-upgraded-downgraded-reference-binary.yml"
