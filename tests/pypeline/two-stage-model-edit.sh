#!/usr/bin/env bash

#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set -euo pipefail

# Drive a two-stage model edit against a running daemon: an invalid intermediate
# must be rejected (Model::deserialize verifies) without crashing the daemon, and
# once the edit is completed the valid model is picked up again.

function log() { echo "$1" > /dev/stderr; }

if [[ $# -ge 1 ]]; then
  BIN_PATH="$(realpath "$1")/bin"
  export PATH="$BIN_PATH:$PATH"
fi

WORKDIR="$(mktemp -d)"
DAEMON_PID=""
cleanup() {
  if [[ -n "$DAEMON_PID" ]]; then
    kill "$DAEMON_PID" 2> /dev/null || true
  fi
  rm -rf "$WORKDIR"
}
trap cleanup EXIT
cd "$WORKDIR"

# A hand-written model: a typedef pointing at a struct. Enough to emit an
# artifact, and the reference can be repointed to break the model.
cat > revng.yml <<'EOF'
---
Architecture: x86_64
DefaultABI: SystemV_x86_64
TypeDefinitions:
  - Kind: StructDefinition
    ID: 1000
    Size: 4
    Fields:
      - Offset: 0
        Type:
          Kind: PrimitiveType
          PrimitiveKind: Unsigned
          Size: 4
  - Kind: TypedefDefinition
    ID: 2000
    Name: my_alias
    UnderlyingType:
      Kind: DefinedType
      Definition: "/TypeDefinitions/1000-StructDefinition"
EOF
cp revng.yml good-model.yml

# Start the daemon in the background, backed by the on-disk project, and wait
# until it answers.
PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
revng project --storage-provider "local://?inline" daemon \
  --bind "127.0.0.1:$PORT" > "$WORKDIR/daemon.log" 2>&1 &
DAEMON_PID=$!

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/status" > /dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$DAEMON_PID" 2> /dev/null; then
    log "the daemon exited during startup:"
    cat "$WORKDIR/daemon.log" > /dev/stderr
    exit 1
  fi
  sleep 1
done

# From now on the client talks to the daemon.
export REVNG_STORAGE_PROVIDER="daemon://127.0.0.1:$PORT"
ARTIFACT=emit-type-and-global-header

# Baseline: the good model produces an artifact.
revng project artifact "$ARTIFACT" > baseline.txt

# --- Stage 1: the first write leaves the model invalid ----------------------
# Repoint the type reference at an id that does not exist: still valid YAML, but
# a model with a dangling reference that fails verification.
sed -E -i 's#(/TypeDefinitions/)[0-9]+(-)#\1999999999\2#g' revng.yml

# The daemon must reject the invalid model, staying up.
if revng project artifact "$ARTIFACT" > /dev/null 2>&1; then
  log "the daemon accepted an invalid model"
  exit 1
fi
if ! kill -0 "$DAEMON_PID" 2> /dev/null; then
  log "the daemon crashed on an invalid model"
  exit 1
fi
if ! grep -qi "verify" "$WORKDIR/daemon.log"; then
  log "the invalid model was rejected for the wrong reason:"
  cat "$WORKDIR/daemon.log" > /dev/stderr
  exit 1
fi

# --- Stage 2: the second write completes the edit, model is valid again -----
cp good-model.yml revng.yml

# The artifact is produced again and matches the baseline: the invalid
# intermediate neither corrupted the daemon nor changed the result.
revng project artifact "$ARTIFACT" > after-valid.txt
if ! diff -u baseline.txt after-valid.txt > /dev/null; then
  log "output changed after an invalid intermediate (daemon state damaged)"
  exit 1
fi
