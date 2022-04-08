#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

MODEL_TMP_FILE=$(mktemp -p "$(pwd)" --suffix=.in.yml)
OUT_MODEL_TMP_FILE=$(mktemp -p "$(pwd)" --suffix=.out.yml)

function cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  rm "$MODEL_TMP_FILE" "$OUT_MODEL_TMP_FILE"
}
trap cleanup SIGINT SIGTERM ERR EXIT

FILE="$2"

revng model opt -Y --verify < "$FILE" > "$MODEL_TMP_FILE"
node - "$MODEL_TMP_FILE" "$OUT_MODEL_TMP_FILE" < "$1"
if ! revng model diff "$MODEL_TMP_FILE" "$OUT_MODEL_TMP_FILE"; then
  echo "Model of $FILE did not reserialize as original"
  exit 1;
fi
