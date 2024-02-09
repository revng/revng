#!/bin/bash
#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

# Python oneliner to get an unbound port to spawn the S3 server on. Note that
# this is not failproof and might fail since there's a time gap between when
# the script is run and the server actually binds the socket
S3_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
# Run the server with the specified port
s3rver --directory "$2" --address 127.0.0.1 --port "$S3_PORT" --configure-bucket test &>/dev/null &
# Save the server pid to kill it on exit
S3RVER_PID=$!
trap 'kill -9 $S3RVER_PID' EXIT

# Run revng with the server as the resume directory
revng artifact \
  --resume="s3://S3RVER:S3RVER@region+127.0.0.1:$S3_PORT/test/project-test-dir" \
  --analyses-list=revng-initial-auto-analysis \
  -o /dev/null \
  disassemble "$1"

# Simple file check, this looks into the persistence directory of s3rver and
# checks that the input file and the file in 'begin/input' have the same
# contents. This is to guarantee that the S3 backend does not change the
# contents of the file.
INDEX_FILE="$2/test/project-test-dir/index.yml._S3rver_object"
test -f "$INDEX_FILE"
INPUT_ID="$(grep -Po '(?<=^begin/input:).*' "$INDEX_FILE" | tr -d ' ' | tr -d "'")"
INPUT_FILE="$2/test/project-test-dir/$INPUT_ID._S3rver_object"
sha256sum --quiet --check <(sha256sum <"$INPUT_FILE") <"$1"
