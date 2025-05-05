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
