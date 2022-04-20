#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

PARSE_MODEL_SCRIPT=$(cat << "EOF"
import sys
import yaml
from revng import model
m = yaml.load(sys.stdin, Loader=model.YamlLoader)
print(yaml.dump(m, Dumper=model.YamlDumper))
EOF
)

function parse_revng_model () {
  python3 -c "$PARSE_MODEL_SCRIPT"
}

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <bitcode-or-yaml-model>"
  exit 1
fi

FILE="$1"

echo "Testing $FILE"
if ! revng model opt -Y --verify < "$FILE" | parse_revng_model | revng model opt -Y --verify; then
  echo "Failed deserializing and reserializing $FILE"
  exit 1;
fi
