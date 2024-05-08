#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

set -euo pipefail

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

# Copy required files from the model module in the test module directory
cp -ar "$SOURCE_ROOT/python/revng/model/metaaddress.py" testmodule
touch testmodule/__init__.py

# Generate python model for v1 and v2
for INDEX in 1 2; do
  "$SOURCE_ROOT/scripts/tuple_tree_generator/tuple-tree-generate-python.py" \
    --namespace dummy \
    --root-type RootType \
    --output "testmodule/v$INDEX/__init__.py" \
    --string-type "string" \
    "$SCRIPT_DIR/v${INDEX}_schema.yml"

  cp "$SOURCE_ROOT/python/revng/model/v1/external.py" "testmodule/v${INDEX}/external.py"
done

export PYTHONPATH="$PWD:$SOURCE_ROOT/python:${PYTHONPATH:+:${PYTHONPATH}}"
cd "$SCRIPT_DIR"
"$SCRIPT_DIR/deserialize_multiple_versions.py"
