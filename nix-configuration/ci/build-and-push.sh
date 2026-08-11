#!/usr/bin/env bash

set +x
set -euo pipefail

log() {
    printf '%s\n' "$*" >&2
}

for REQUIRED_VARIABLE in \
    REVNG_CACHE_SSH_TARGET \
    REVNG_CACHE_PUBLIC_SIGNING_KEY \
    REVNG_CACHE_PRIVATE_SIGNING_KEY \
    REVNG_CACHE_SSH_PRIVATE_KEY \
    REVNG_CACHE_SSH_KNOWN_HOSTS
do
    if [ -z "${!REQUIRED_VARIABLE:-}" ]; then
        log "Required environment variable ${REQUIRED_VARIABLE} is missing or empty."
        exit 1
    fi
done
unset REQUIRED_VARIABLE

umask 077
CREDENTIAL_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/revng-cache-push.XXXXXXXXXX")"
chmod 700 "$CREDENTIAL_DIRECTORY"

PUBLIC_SIGNING_KEY_FILE="$CREDENTIAL_DIRECTORY/public-signing-key"
PRIVATE_SIGNING_KEY_FILE="$CREDENTIAL_DIRECTORY/private-signing-key"
SSH_PRIVATE_KEY_FILE="$CREDENTIAL_DIRECTORY/ssh-private-key"
SSH_KNOWN_HOSTS_FILE="$CREDENTIAL_DIRECTORY/ssh-known-hosts"
UPLOADER_CONFIG_FILE="$CREDENTIAL_DIRECTORY/nix-cache-push.yml"

cleanup() {
    rm -f -- \
        "$PUBLIC_SIGNING_KEY_FILE" \
        "$PRIVATE_SIGNING_KEY_FILE" \
        "$SSH_PRIVATE_KEY_FILE" \
        "$SSH_KNOWN_HOSTS_FILE" \
        "$UPLOADER_CONFIG_FILE"
    rmdir -- "$CREDENTIAL_DIRECTORY" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s\n' "$REVNG_CACHE_PUBLIC_SIGNING_KEY" > "$PUBLIC_SIGNING_KEY_FILE"
printf '%s\n' "$REVNG_CACHE_PRIVATE_SIGNING_KEY" > "$PRIVATE_SIGNING_KEY_FILE"
printf '%s\n' "$REVNG_CACHE_SSH_PRIVATE_KEY" > "$SSH_PRIVATE_KEY_FILE"
printf '%s\n' "$REVNG_CACHE_SSH_KNOWN_HOSTS" > "$SSH_KNOWN_HOSTS_FILE"
chmod 600 \
    "$PUBLIC_SIGNING_KEY_FILE" \
    "$PRIVATE_SIGNING_KEY_FILE" \
    "$SSH_PRIVATE_KEY_FILE" \
    "$SSH_KNOWN_HOSTS_FILE"
unset REVNG_CACHE_PUBLIC_SIGNING_KEY
unset REVNG_CACHE_PRIVATE_SIGNING_KEY
unset REVNG_CACHE_SSH_PRIVATE_KEY
unset REVNG_CACHE_SSH_KNOWN_HOSTS

cat > "$UPLOADER_CONFIG_FILE" <<EOF
default: public
caches:
  public:
    target: ${REVNG_CACHE_SSH_TARGET}:/public
    secret_key_file: $PUBLIC_SIGNING_KEY_FILE
  private:
    target: ${REVNG_CACHE_SSH_TARGET}:/private
    secret_key_file: $PRIVATE_SIGNING_KEY_FILE
EOF
chmod 600 "$UPLOADER_CONFIG_FILE"

printf -v RSYNC_RSH \
    'ssh -i %q -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=%q -o GlobalKnownHostsFile=/dev/null -o UpdateHostKeys=no' \
    "$SSH_PRIVATE_KEY_FILE" \
    "$SSH_KNOWN_HOSTS_FILE"
export RSYNC_RSH

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIRECTORY/../.." && pwd)"
REPOSITORY_FLAKE="$REPOSITORY_ROOT"
cd "$REPOSITORY_ROOT"

export REVNG_NIX_NONINTERACTIVE=1
export REVNG_NIX_STATE="${REVNG_NIX_STATE:-local}"

./nix init
./nix build --no-link \
    "${REPOSITORY_FLAKE}#revng" \
    "${REPOSITORY_FLAKE}#test/revng"

./nix run "${REPOSITORY_FLAKE}#nix-cache-push" -- push \
    --config "$UPLOADER_CONFIG_FILE" \
    "${REPOSITORY_FLAKE}#revng" \
    "${REPOSITORY_FLAKE}#test/revng"
