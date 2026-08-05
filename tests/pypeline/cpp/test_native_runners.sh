#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

BIN_PATH="$(realpath "$1")/bin"
export PATH="$BIN_PATH:$PATH"

DEBUG_DIR=$(mktemp -d --tmpdir tmp.revng-pypeline-native-debug-test.XXXXXXXXXX)
trap 'rm -rf -- "$DEBUG_DIR"' EXIT

revng quick artifact output -o /dev/null --debug "$DEBUG_DIR" /dev/null

# Assert that the debug dir is as it is expected
cd "$DEBUG_DIR"
[[ -f "0000-CreateEntryPipe/run" ]]
[[ -f "0001-AppendFooLibAnalysis2/run" ]]
[[ -f "0002-CreateEntryPipe/run" ]]
[[ -f "0003-AppendFooPipe2/run" ]]

# Check analysis result
[[ $(yq -r '.ImportedLibraries[0]' 0001-AppendFooLibAnalysis2/output_revng.yml) == "foo.so" ]]

# Check pipe result
function tar_output() {
    local DIRECTORY="$1"
    shift;
    tar -f "$DIRECTORY/outputs/container.tar" "$@" 2>/dev/null
}

[[ $(tar_output 0000-CreateEntryPipe -t) == "/binary" ]]
[[ $(tar_output 0002-CreateEntryPipe -t) == "/binary" ]]
[[ $(tar_output 0003-AppendFooPipe2 -t) == "/binary" ]]

[[ $(tar_output 0000-CreateEntryPipe -x --to-stdout "/binary") == "foo" ]]
[[ $(tar_output 0002-CreateEntryPipe -x --to-stdout "/binary") == "foo" ]]
[[ $(tar_output 0003-AppendFooPipe2 -x --to-stdout "/binary") == "foofoo" ]]
