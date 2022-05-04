#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

OUTPUT="$PWD/MultiStepPipelineOut.txt"
WORKING_DIRECTORY="$PWD/MultiStepPipelineTestDir/"
REFERENCE_OUTPUT="$SCRIPT_DIR/MultiStepPipelineOutput.txt"

function run() {
  "$PWD/libexec/revng/revng-pipeline" \
    -P="$SCRIPT_DIR/MultiStepPipeline.yml" \
    SecondStep:Strings3:Root:StringKind\
    -i "begin:Strings1:$SCRIPT_DIR/MultiStepPipelineInput.txt" \
    -o "SecondStep:Strings3:$OUTPUT" \
    -p "$WORKING_DIRECTORY" \
    "$@"
}

# Cleanup
rm -rf "$WORKING_DIRECTORY"
mkdir -p "$WORKING_DIRECTORY"

# Produce $OUTPUT
run "$@"
diff "$OUTPUT" "$REFERENCE_OUTPUT"

# Produce $OUTPUT again and make sure it has not changed
run "$@"
diff "$OUTPUT" "$REFERENCE_OUTPUT"

# Drop an intermediate results and recompute
rm "$WORKING_DIRECTORY/SecondStep/Strings1"
run "$@"
diff "$OUTPUT" "$REFERENCE_OUTPUT"

# Recreate the output again
rm "$OUTPUT"
run "$@"
diff "$OUTPUT" "$REFERENCE_OUTPUT"
