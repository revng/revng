#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

WORKDIR=$(mktemp --tmpdir -d tmp.revng-pypeline-compare.XXXXXXXXXX)
trap 'rm -r "$WORKDIR"' EXIT

MODEL="$1"
BINARY="$2"
ARTIFACT="$3"

OLD_PATH="$WORKDIR/old"
NEW_PATH="$WORKDIR/new"
mkdir "$WORKDIR/cache"
export REVNG_CACHE_DIR="$WORKDIR/cache"


function files_in_dir() {
    find "$1" -type f -printf '%f\n'
}

# JQ appends a newline '\n' to the output, this command runs it and strips it
function jq_exact() {
    jq "$@" | head -c -1
}

# Comparison between multiple artifact objects (e.g. function, type-definition)
function compare() {
    local PREFIX="$1"
    local OLD_EXTRACTED NEW_EXTRACTED OLD_FILES
    OLD_EXTRACTED="$WORKDIR/old_extracted"
    NEW_EXTRACTED="$WORKDIR/new_extracted"
    mkdir "$OLD_EXTRACTED" "$NEW_EXTRACTED"

    # Extract all the files from the tar (old format), rename them without all
    # the extensions
    tar -C"$OLD_EXTRACTED" -xf "$OLD_PATH"
    readarray -t OLD_FILES < <(files_in_dir "$OLD_EXTRACTED")

    local FILE
    for FILE in "${OLD_FILES[@]}"; do
        local KEY_EXTRACTED
        KEY_EXTRACTED=$(cut -d. -f1 <<< "$FILE")
        mv "$OLD_EXTRACTED/$FILE" "$OLD_EXTRACTED/$KEY_EXTRACTED"
    done

    # Extract all the files from the JSON (new format), strip away the prefix
    # from each entry
    local KEY
    jq -r '. | keys | .[]' "$NEW_PATH" | \
    while IFS= read -r KEY; do
        local KEY_EXTRACTED="${KEY/#$PREFIX/}"
        jq_exact -r ".[\"$KEY\"]" "$NEW_PATH" > "$NEW_EXTRACTED/$KEY_EXTRACTED"
    done

    if ! diff <(files_in_dir "$OLD_EXTRACTED" | sort) <(files_in_dir "$NEW_EXTRACTED" | sort); then
        echo "File list mismatch!"
        return 1
    fi

    # Compare the files for each element
    local OK=0
    while IFS= read -r KEY; do
        local DIFF_OUTPUT RC=0
        DIFF_OUTPUT=$(mktemp --tmpdir="$WORKDIR")

        diff -u "$OLD_EXTRACTED/$KEY" "$NEW_EXTRACTED/$KEY" > "$DIFF_OUTPUT" || RC=$?
        if [[ "$RC" -ne 0 ]]; then
            OK=1
            echo "Comparison for $ARTIFACT: $KEY failed"
            cat "$DIFF_OUTPUT"
        fi
    done < <(files_in_dir "$OLD_EXTRACTED")

    return "$OK"
}


# Comparison of the `lift` artifact, this is special because the two formats
# are quite different and need to be converted to `.ll` before comparing
function compare_lift() {
    # Convert old
    zstdcat "$OLD_PATH" | revng opt -S > "$WORKDIR/old.ll"
    # Convert new
    jq_exact -r '.["/binary"]' "$NEW_PATH" | base64 -d | revng opt -S > "$WORKDIR/new.ll"

    local DIFF_OUTPUT RC=0
    DIFF_OUTPUT=$(mktemp --tmpdir="$WORKDIR")
    diff -u "$WORKDIR/old.ll" "$WORKDIR/new.ll" > "$DIFF_OUTPUT" || RC=$?
    if [[ "$RC" -ne 0 ]]; then
        echo "Comparison for $ARTIFACT failed!"
        cat "$DIFF_OUTPUT"
    fi

    return "$RC"
}

# Comparison between single-valued artifacts
function compare_binary() {
    local DIFF_OUTPUT RC=0
    DIFF_OUTPUT=$(mktemp --tmpdir="$WORKDIR")

    diff -u "$OLD_PATH" <(jq_exact -r '.["/binary"]' "$NEW_PATH") > "$DIFF_OUTPUT" || RC=$?
    if [[ "$RC" -ne 0 ]]; then
        echo "Comparison for $ARTIFACT failed!"
        cat "$DIFF_OUTPUT"
    fi

    return "$RC"
}

# Compute the artifacts from both the old and new pipeline
revng artifact "$ARTIFACT" --model="$MODEL" "$BINARY" -o "$OLD_PATH"
revng2 project artifact "$ARTIFACT" -o "$NEW_PATH" 2>/dev/null

RC=0
# NOTE: this could be derived from `pipeline-description.yml`, but since this
#       is temporary they are hardcorded here.
if [[ "$ARTIFACT" = "disassemble" ]]; then
    compare "/function/" || RC=$?
elif [[ "$ARTIFACT" = "emit-type-definitions" ]]; then
    compare "/type-definition/" || RC=$?
elif [[ "$ARTIFACT" = "lift" ]]; then
    compare_lift || RC=$?
else
    compare_binary || RC=$?
fi

if [[ "$RC" -ne 0 ]]; then
    echo "Comparison for $ARTIFACT failed!"
fi

exit "$RC"
