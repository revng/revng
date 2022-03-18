#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <dir-with-binaries> <output-dir>"
  exit 1
fi

SOURCE_BINARIES="$1"
OUTPUT_DIR="$2"

if [[ ! -d "$SOURCE_BINARIES" ]]; then
  echo "$SOURCE_BINARIES is not a directory"
  exit 1
fi

find "$SOURCE_BINARIES" -type f -executable |
(while read -r BINARY; do
  echo "Converting $BINARY"

  if file "$BINARY" | grep "64-bit"; then
    IDA=ida64
    FORMAT=i64
  else
    IDA=ida
    FORMAT=idb
  fi

  OUTPUT="$OUTPUT_DIR/$(basename "$BINARY").$FORMAT"
  echo "Saving to $OUTPUT"

  if ! $IDA -A -B -o  "$BINARY"; then
    echo "Failed converting $BINARY"
    exit 1
  fi
done)
