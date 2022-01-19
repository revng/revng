#!/bin/bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -eux
set -o pipefail

ABI_NAME="$1"
WORKING_DIRECTORY="$2"
RUNTIME_ABI_ANALYSIS_RESULT="$3"
LIFTED_REFERENCE_BINARY="$4"

mkdir -p "${WORKING_DIRECTORY}/model"
revng model dump "${LIFTED_REFERENCE_BINARY}" -o="${WORKING_DIRECTORY}/model/reference_binary.yml"

revng model opt --convert-all-cabi-functions-to-raw --abi="${ABI_NAME}" "${WORKING_DIRECTORY}/model/reference_binary.yml" -o="${WORKING_DIRECTORY}/model/downgraded_reference_binary.yml"
revng model opt --convert-all-raw-functions-to-cabi --abi="${ABI_NAME}" "${WORKING_DIRECTORY}/model/downgraded_reference_binary.yml" -o="${WORKING_DIRECTORY}/model/upgraded_downgraded_reference_binary.yml"
revng model opt --convert-all-cabi-functions-to-raw --abi="${ABI_NAME}" "${WORKING_DIRECTORY}/model/upgraded_downgraded_reference_binary.yml" -o="${WORKING_DIRECTORY}/model/downgraded_upgraded_downgraded_reference_binary.yml"

revng abi-verify -abi="${ABI_NAME}" "${WORKING_DIRECTORY}/model/downgraded_reference_binary.yml" "${RUNTIME_ABI_ANALYSIS_RESULT}"

revng model diff "${WORKING_DIRECTORY}/model/downgraded_reference_binary.yml" "${WORKING_DIRECTORY}/model/downgraded_upgraded_downgraded_reference_binary.yml"
