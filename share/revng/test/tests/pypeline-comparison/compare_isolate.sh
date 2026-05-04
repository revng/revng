#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

WORKDIR=$(mktemp --tmpdir -d tmp.revng-pypeline-compare.XXXXXXXXXX)
trap 'rm -r "$WORKDIR"' EXIT

MODEL="$(realpath "$1")"
BINARY="$(realpath "$2")"

mkdir "$WORKDIR/cache" "$WORKDIR/old" "$WORKDIR/new"
export XDG_CACHE_HOME="$WORKDIR/cache"
cp -raT "$PWD" "$WORKDIR/scratch"
cd "$WORKDIR/scratch"

# This function takes a bitcode file and normalizes it a bit, with the following:
# * extract just the isolated function with `llvm-extract`, converting the
#   other functions to declarations
# * Run it through globaldce and convert it to textual IR
# * All metadata IDs are blanked to '!0'
# * Metadata declarations are dropped
# * Empty lines and lines starting with comments are dropped
function normalize() {
    local FUNCTION="$1"
    FUNCTION="${FUNCTION//:/_}"
    llvm-extract --func="local_$FUNCTION" | revng opt -globaldce -S | \
        sed -e 's;![0-9]\+;!0;g' -e 's;#[0-9]\+;#0;g' | \
        grep -v -e '^!0 = ' -e '^\s*$' -e '^;' -e '^declare' -e '^attributes #0 ='
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

    revng artifact isolate --model="$MODEL" "$BINARY" "$FUNCTION" | zstdcat | \
        normalize "$FUNCTION" > "$WORKDIR/old/$FUNCTION.ll"

    OBJECT_ID="/function/$FUNCTION"
    revng2 project artifact isolate --format yaml "$OBJECT_ID" 2>/dev/null | \
        yq -r ".[\"/function/$FUNCTION\"]" | base64 -d | \
        normalize "$FUNCTION" > "$WORKDIR/new/$FUNCTION.ll"

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
