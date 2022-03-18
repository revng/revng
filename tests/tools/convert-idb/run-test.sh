#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

find "$SCRIPT_DIR" -name "*.i64"  |
(while read -r IDB; do
  echo "Converting $IDB"
  if ! revng model import idb "$IDB" | revng model opt -Y --verify > /dev/null; then
    echo "Test failed on IDB $IDB"
    exit 1
  fi
done)

echo "Tests passed"
