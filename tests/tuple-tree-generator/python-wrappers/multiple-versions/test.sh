#!/bin/bash

set -e
set -o pipefail

function log() {
  echo "$1" > /dev/stderr
}

SOURCE_ROOT="$1"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if ! test -d "$SOURCE_ROOT"; then
  log "Usage: $0 SOURCE_ROOT"
  exit 1
fi

rm -rf testmodule

mkdir testmodule
mkdir testmodule/v1
mkdir testmodule/v2

# Copy the model._common directory in the build directory
cp -ar "$SOURCE_ROOT/python/revng/model/_common" testmodule

# Copy the base class
cp -a "$SCRIPT_DIR/base.py" testmodule/v1/
cp -a "$SCRIPT_DIR/base.py" testmodule/v2/

# Generate python model for v1 and v2
for INDEX in 1 2; do
  datamodel-codegen \
    --base-class .base.MonkeyPatchingBaseClass \
    --target-python-version 3.6 \
    --input "$SCRIPT_DIR/v""$INDEX""_schema.yml" \
    > "testmodule/v""$INDEX""/_generated.py"
done

export PYTHONPATH="$PWD:${PYTHONPATH:+:${PYTHONPATH}}"
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/deserialize_multiple_versions.py"
