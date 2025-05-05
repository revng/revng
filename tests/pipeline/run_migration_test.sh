#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
set -euo pipefail

log() {
  echo "$@" >&2
}

log_file() {
  cat "$@" >&2
}

TEST_DIR=$(mktemp -d)

trap 'rm --recursive --force "${TEST_DIR}"' EXIT

mkdir --parents "${TEST_DIR}"/resume-dir/context

#
# version 1 -> 3
#

# TODO: consider splitting different versions into different files.

cat > "${TEST_DIR}"/resume-dir/context/model.yml <<EOF
Version: 1
Configuration:
  Disassembly:
   UseATTSyntax: true
EOF

./bin/revng artifact --resume "${TEST_DIR}"/resume-dir emit-model-header /dev/null

USE_ATT_SYNTAX=$(yq '.Configuration.Disassembly.UseATTSyntax' "${TEST_DIR}"/resume-dir/context/model.yml)
USE_X86_ATT_SYNTAX=$(yq '.Configuration.Disassembly.UseX86ATTSyntax' "${TEST_DIR}"/resume-dir/context/model.yml)

if [ "${USE_ATT_SYNTAX}" != "null" ]; then
  log "Expected UseATTSyntax = null, but got ${USE_ATT_SYNTAX} instead"
  log "Here's the migrated file for reference"
  log_file "${TEST_DIR}"/resume-dir/context/model.yml
  exit 1
fi

if [ "${USE_X86_ATT_SYNTAX}" != "true" ]; then
  log "Expected UseX86ATTSyntax = true, but got ${USE_X86_ATT_SYNTAX} instead"
  log "Here's the migrated file for reference"
  log_file "${TEST_DIR}"/resume-dir/context/model.yml
  exit 1
fi

#
# version 3 -> 4
#

# TODO: consider splitting different versions into different files.

cat > "${TEST_DIR}"/resume-dir/context/model.yml <<EOF
Version:         3
Functions:
  - Entry:           "0x1000:Code_aarch64"
    Comments:
      - Location:
          - "0x1000:Code_aarch64"
        Body:            test_0
      - Location:
          - "0x1000:Code_aarch64"
        Body:            test_1
      - Location:
          - "0x1004:Code_aarch64"
        Body:            test_2

# All of the following is just here to ensure the basic model above verifies
Segments:
  - StartAddress:    "0x1000:Generic64"
    VirtualSize:     256
    StartOffset:     0
    FileSize:        256
    IsReadable:      false
    IsWriteable:     false
    IsExecutable:    true
DefaultPrototype:
  Kind:            DefinedType
  Definition:      "/TypeDefinitions/0-RawFunctionDefinition"
TypeDefinitions:
  - ID:              0
    Kind:            RawFunctionDefinition
    Architecture:    aarch64
Architecture:    aarch64
EOF

./bin/revng artifact --resume "${TEST_DIR}"/resume-dir emit-model-header /dev/null

TEST_0_INDEX=$(yq '.Functions[0].Comments[0].Index' "${TEST_DIR}"/resume-dir/context/model.yml)
TEST_1_INDEX=$(yq '.Functions[0].Comments[1].Index' "${TEST_DIR}"/resume-dir/context/model.yml)
TEST_2_INDEX=$(yq '.Functions[0].Comments[2].Index' "${TEST_DIR}"/resume-dir/context/model.yml)

if [[ "${TEST_0_INDEX}" != "0" || "${TEST_1_INDEX}" != "1" || "${TEST_2_INDEX}" != "2" ]]; then
  log "Expected all the comment indices to be consistent after the migration"
  log "Here's the migrated file for reference"
  log_file "${TEST_DIR}"/resume-dir/context/model.yml
  exit 1
fi
