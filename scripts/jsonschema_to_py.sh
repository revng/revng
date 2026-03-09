#!/usr/bin/env bash
set -euo pipefail

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# This script converts a jsonschema json/yaml file to a python file with
# TypedDict declarations, useful for having typing hints and checks on the data
# represented by the schema.

INPUT="$1"
OUTPUT="$2"
ROOT_NAME="$3"

# The python package only works with JSON, use `yq` to normalize both YAML
# and JSON inputs into JSON
python -m jsonschema_to_typeddict \
    --output-path /dev/stdout \
    --root-name "$ROOT_NAME" \
    <(yq . "$INPUT") | \
# The template used by the python package is a bit ugly and uses
# `typing_extensions` to be compatible with old python versions. Use `sed` to
# make the output look a bit nicer.
sed \
    -e 's;typ\.;;' \
    -e 's;import typing_extensions as typ;from typing import Literal, TypedDict, TypeAlias;' \
> "$OUTPUT"
