#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

WORKDIR=$(mktemp --tmpdir -d tmp.revng-pypeline-compare.XXXXXXXXXX)
trap 'rm -r "$WORKDIR"' EXIT

MODEL="$1"
BINARY="$2"

mkdir "$WORKDIR/cache" "$WORKDIR/old" "$WORKDIR/new"
export REVNG_CACHE_DIR="$WORKDIR/cache"

# This function takes a bitcode file and normalizes it a bit, with the following:
# * Run it through globaldce and convert it to textual IR
# * All metadata IDs are blanked to '!0'
# * Metadata declarations are dropped
# * Empty lines and lines starting with comments are dropped
function normalize() {
    revng opt -globaldce -S | sed 's;![0-9]\+;!0;g' | grep -v -e '^!0 = ' -e '^\s*$' -e '^;'
}

OK=0
while IFS= read -r FUNCTION; do
    # When working with an LLVMContext, types are pooled into the context. This
    # poses a problem when adding a type with the same name twice; to fix this
    # the context automatically appends a `.[0-9]+` suffix to these to
    # disambiguate. Since this test relies on the diff being identical, both
    # old and new pipeline commands need to be run individually on each
    # function, otherwise the types will have the numeric suffix and the
    # comparison will fail.

    revng artifact isolate --model="$MODEL" "$BINARY" "$FUNCTION" | \
        zstdcat | normalize > "$WORKDIR/old/$FUNCTION.ll"

    OBJECT_ID="/function/$FUNCTION"
    revng2 project artifact isolate "$OBJECT_ID" 2>/dev/null | \
        jq -r ".[\"/function/$FUNCTION\"]" | base64 -d | normalize > "$WORKDIR/new/$FUNCTION.ll"

    DIFF_OUTPUT="$WORKDIR/diff_output_$FUNCTION"
    RC=0
    diff -u "$WORKDIR/old/$FUNCTION.ll" "$WORKDIR/new/$FUNCTION.ll" > "$DIFF_OUTPUT" || RC=$?
    if [[ "$RC" -ne 0 ]]; then
        echo "Comparison failed for $FUNCTION"
        cat "$DIFF_OUTPUT"
        OK=1
    fi
done < <(yq -r '.Functions[].Entry' "$MODEL")

exit "$OK"
