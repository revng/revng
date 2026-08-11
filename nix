#!/usr/bin/env bash

# Self-contained nix wrapper for the revng repo. On first use, the
# user runs `./nix init` to download a pinned static nix binary and
# wire up an isolated nix.conf. Interactive initialization prompts for
# a GitLab token to authenticate to the private rev.ng cache. Every subsequent
# invocation just execs the static nix with the isolated config; if
# init has not been run, the script prints instructions and exits 1.
#
# State (nix binary, isolated store, config, netrc) lives either in a
# shared XDG cache directory (default: $XDG_CACHE_HOME/revng/nix, i.e.
# ~/.cache/revng/nix), so multiple checkouts of the repo share the
# same store, or in ./.nix next to this script, local to the checkout.
# `nix init` asks which one to use. For unattended setup,
# REVNG_NIX_STATE=shared|local selects it and
# REVNG_NIX_NONINTERACTIVE=1 disables prompts. REVNG_NIX_CACHE_TOKEN
# optionally enables private-cache reads. Delete the picked directory
# to reset.
#
# Sandboxing: nix 2.10+ automatically switches to a chroot store when
# /nix/store is not accessible, using unprivileged user + mount
# namespaces internally. This wrapper requires the host kernel to
# allow them (default on nixpkgs-based systems and Ubuntu 24.04+;
# Debian ≤11 and some hardened kernels disable via
# `sysctl kernel.unprivileged_userns_clone=0`). The script refuses to
# run if the check fails and prints the sysctl to enable them.
#
# The static nix binary comes from nixie-dev/nixie's build cache
# (nix-wrap.cachix.org). To bump the version, on any machine with nix:
#   nix eval --raw github:nixie-dev/nixie#static-bins.outPath
# and paste the store-hash prefix (the 32 chars before the first '-')
# into NIX_BINS_HASH below; then bump NIX_VERSION accordingly.
#
# Example:
#   ./nix init                  # one-time setup
#   ./nix build .#revng -j0     # substitute the closure, no local build

set -euo pipefail

# Logging helper: writes to stderr, so stdout stays clean for whatever
# nix produces. Accepts either arguments or stdin (for multi-line
# heredocs).
log() {
    if [ "$#" -eq 0 ]; then
        cat >&2
    else
        printf '%s\n' "$*" >&2
    fi
}

# Detect whether unprivileged user namespaces work on this host. Tries
# the functional test first (util-linux's `unshare`), falls back to
# the two sysctls kernels commonly gate userns behind. Returns 0 iff
# the host can create user namespaces without root.
check_user_namespaces() {
    if command -v unshare >/dev/null 2>&1; then
        unshare -Ur true 2>/dev/null && return 0
        return 1
    fi
    if [ -r /proc/sys/user/max_user_namespaces ] \
        && [ "$(cat /proc/sys/user/max_user_namespaces)" = 0 ]; then
        return 1
    fi
    if [ -r /proc/sys/kernel/unprivileged_userns_clone ] \
        && [ "$(cat /proc/sys/kernel/unprivileged_userns_clone)" = 0 ]; then
        return 1
    fi
    return 0
}

if ! check_user_namespaces; then
    log <<EOF
Unprivileged user namespaces are disabled on this host. The static nix
binary uses them to run its chroot store without root privileges;
without them, every nix command will fail.

Enable them, as root:

    sysctl -w kernel.unprivileged_userns_clone=1
    sysctl -w user.max_user_namespaces=10000

Or make the change persistent:

    printf '%s\n' 'kernel.unprivileged_userns_clone=1' \\
                  'user.max_user_namespaces=10000' \\
        | sudo tee /etc/sysctl.d/99-userns.conf
    sudo sysctl --system

Then rerun.
EOF
    exit 1
fi

# Pinned static nix, hosted by nixie on nix-wrap.cachix.org. See the
# header comment for the bump procedure.
NIX_VERSION="${NIX_VERSION:-2.26.2}"
NIX_BINS_HASH="${NIX_BINS_HASH:-fp03kid1238djnac7295kfj04vvdg0k4}"
NIX_ARCHITECTURE="$(uname -m)"
NIX_URL="${NIX_URL:-https://nix-wrap.cachix.org/serve/${NIX_BINS_HASH}/nix.Linux.${NIX_ARCHITECTURE}}"

# Candidate state directories.
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XDG_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
CACHE_DIRECTORY="$XDG_CACHE_ROOT/revng/nix"
LOCAL_DIRECTORY="$SCRIPT_DIRECTORY/.nix"

# A completed init leaves this marker plus the static nix binary in
# the chosen directory. Their joint presence in exactly one of the
# two candidates tells subsequent invocations which one won; a lone
# marker (e.g. from a prior wrapper generation) is treated as
# uninitialized.
CACHE_MARKER="$CACHE_DIRECTORY/.setup-complete"
LOCAL_MARKER="$LOCAL_DIRECTORY/.setup-complete"
CACHE_BINARY="$CACHE_DIRECTORY/nix-static"
LOCAL_BINARY="$LOCAL_DIRECTORY/nix-static"

# Validate the unattended-init controls once, before either init or a
# regular command uses them.
case "${REVNG_NIX_NONINTERACTIVE:-0}" in
    0) NONINTERACTIVE=0 ;;
    1) NONINTERACTIVE=1 ;;
    *)
        log "REVNG_NIX_NONINTERACTIVE must be 0 or 1."
        exit 1
        ;;
esac

case "${REVNG_NIX_STATE:-}" in
    ""|shared|local) ;;
    *)
        log "REVNG_NIX_STATE must be 'shared' or 'local'."
        exit 1
        ;;
esac

state_directory_from_environment() {
    case "${REVNG_NIX_STATE:-}" in
        shared) printf '%s\n' "$CACHE_DIRECTORY" ;;
        local)  printf '%s\n' "$LOCAL_DIRECTORY" ;;
        "")     return 1 ;;
    esac
}

# rev.ng caches.
GATING_PROJECT_URL="https://rev.ng/gitlab/revng-private/binary-archives"

TOKEN_URL="${GATING_PROJECT_URL}/-/settings/access_tokens"

PUBLIC_CACHE_URL="https://rev.ng/nix-binary-cache/public/"
PRIVATE_CACHE_URL="https://rev.ng/nix-binary-cache/private/"
REVNG_PUBLIC_KEY="revng-cache:Wqy0YTHRGuDijpuHK+3uhP54idwTYbjXfxVqnsfusGU="
NIXOS_PUBLIC_KEY="cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="

run_init() {
    local NIX_DIRECTORY
    if NIX_DIRECTORY="$(state_directory_from_environment)"; then
        :
    elif [ -f "$CACHE_MARKER" ] && [ -x "$CACHE_BINARY" ]; then
        NIX_DIRECTORY="$CACHE_DIRECTORY"
    elif [ -f "$LOCAL_MARKER" ] && [ -x "$LOCAL_BINARY" ]; then
        NIX_DIRECTORY="$LOCAL_DIRECTORY"
    elif [ "$NONINTERACTIVE" = 1 ]; then
        # Match the interactive prompt's default.
        NIX_DIRECTORY="$CACHE_DIRECTORY"
    else
        log <<EOF
Where should the nix state directory live?

  1) $CACHE_DIRECTORY
     Shared across all revng checkouts (recommended if you have more
     than one checkout — you download nix and populate the store once).

  2) $LOCAL_DIRECTORY
     Scoped to this checkout only.

EOF
        read -r -p "Choose [1/2] (default 1): " ANSWER
        case "${ANSWER:-1}" in
            1|"") NIX_DIRECTORY="$CACHE_DIRECTORY" ;;
            2)    NIX_DIRECTORY="$LOCAL_DIRECTORY" ;;
            *)    log "Invalid choice, aborting."; exit 1 ;;
        esac
    fi

    local NIX_BINARY="$NIX_DIRECTORY/nix-static"
    local CONFIG_FILE="$NIX_DIRECTORY/nix.conf"
    local NETRC_FILE="$NIX_DIRECTORY/netrc"
    local SETUP_COMPLETE_MARKER="$NIX_DIRECTORY/.setup-complete"
    local ALREADY_INITIALIZED=0

    if [ -f "$SETUP_COMPLETE_MARKER" ] && [ -x "$NIX_BINARY" ]; then
        ALREADY_INITIALIZED=1
        if [ -z "${REVNG_NIX_CACHE_TOKEN:-}" ]; then
            log "Already initialized in $NIX_DIRECTORY."
            log "Keeping its existing cache configuration and credentials."
            return 0
        fi
        log "Updating the private-cache credential in $NIX_DIRECTORY."
    fi

    mkdir -p "$NIX_DIRECTORY"

    # Skip the download if a previous init made it this far and then
    # failed at the sanity check.
    if [ ! -x "$NIX_BINARY" ]; then
        log "Downloading pinned static nix ${NIX_VERSION} (${NIX_ARCHITECTURE}) into ${NIX_DIRECTORY}"
        curl -fsSL "$NIX_URL" -o "$NIX_BINARY.tmp"
        chmod +x "$NIX_BINARY.tmp"
        mv "$NIX_BINARY.tmp" "$NIX_BINARY"
    fi

    # Token / netrc bootstrap.
    #
    # Nix's HTTP binary-cache client authenticates via netrc, i.e. HTTP
    # Basic. GitLab rejects Basic on its REST API but accepts a token
    # as the Basic password on git-over-HTTPS, which is what the
    # private-cache nginx auth_request forwards to. The token goes in
    # the password field; any non-empty login works.
    #
    # The token is OPTIONAL: skipping it configures the public cache
    # only, which is enough if the closures you build don't reference
    # any `fetchPrivateUrl`-produced paths (Windows SDKs, non-
    # redistributable tarballs, and so on).
    local TOKEN=""
    if [ -n "${REVNG_NIX_CACHE_TOKEN:-}" ]; then
        TOKEN="$REVNG_NIX_CACHE_TOKEN"
    elif [ "$NONINTERACTIVE" = 0 ]; then
        log <<EOF

The private cache is gated on read access to
  ${GATING_PROJECT_URL}

Create a project access token at:

  ${TOKEN_URL}

  Name:   nix-binary-cache
  Role:   Reporter
  Scopes: read_repository

Leave the prompt empty to configure the public cache only — that is
enough unless you build something that pulls a private FOD (e.g. a
Windows SDK).

EOF
        read -r -s -p "Paste the token (input hidden, or ENTER to skip): " TOKEN
        log ""
    fi

    # Sanity-check before replacing any existing config, so a bad
    # replacement token cannot destroy a working initialization.
    log ""
    log "Sanity check: nix-cache-info on the configured cache(s)."
    local PUBLIC_CODE PRIVATE_CODE CHECK_FAILED=0
    PUBLIC_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -m 15 "${PUBLIC_CACHE_URL}nix-cache-info" || echo error)
    log "    public   -> $PUBLIC_CODE"
    [ "$PUBLIC_CODE" != 200 ] && CHECK_FAILED=1

    if [ -n "$TOKEN" ]; then
        PRIVATE_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -m 15 -u "nix-cache:$TOKEN" "${PRIVATE_CACHE_URL}nix-cache-info" || echo error)
        log "    private  -> $PRIVATE_CODE"
        [ "$PRIVATE_CODE" != 200 ] && CHECK_FAILED=1
    else
        log "    private  -> skipped (no token)"
    fi

    if [ "$CHECK_FAILED" != 0 ]; then
        log "Sanity check failed. Rerun \`$(basename "$0") init\` to retry."
        if [ -n "${PRIVATE_CODE:-}" ] && { [ "$PRIVATE_CODE" = 401 ] || [ "$PRIVATE_CODE" = 403 ]; }; then
            log "Private returned $PRIVATE_CODE: check that the token has read_repository"
            log "access on ${GATING_PROJECT_URL} (Reporter role or higher)."
        fi

        exit 1
    fi

    # nix.conf, plus optional netrc: two shapes depending on whether
    # the private cache is wired in. Write temporary files first so
    # an interrupted credential replacement leaves the old setup
    # usable.
    local CONFIG_TEMP="$CONFIG_FILE.tmp.$$"
    local NETRC_TEMP="$NETRC_FILE.tmp.$$"
    if [ -n "$TOKEN" ]; then
        ( umask 077
          cat > "$NETRC_TEMP" <<EOF
machine rev.ng
  login nix-cache
  password $TOKEN
EOF
        )
        chmod 600 "$NETRC_TEMP"
        cat > "$CONFIG_TEMP" <<EOF
experimental-features = nix-command flakes
substituters = ${PUBLIC_CACHE_URL} ${PRIVATE_CACHE_URL} https://cache.nixos.org/
trusted-public-keys = ${REVNG_PUBLIC_KEY} ${NIXOS_PUBLIC_KEY}
netrc-file = ${NETRC_FILE}
EOF
        mv "$NETRC_TEMP" "$NETRC_FILE"
    else
        cat > "$CONFIG_TEMP" <<EOF
experimental-features = nix-command flakes
substituters = ${PUBLIC_CACHE_URL} https://cache.nixos.org/
trusted-public-keys = ${REVNG_PUBLIC_KEY} ${NIXOS_PUBLIC_KEY}
EOF
        rm -f "$NETRC_FILE"
    fi
    mv "$CONFIG_TEMP" "$CONFIG_FILE"

    touch "$SETUP_COMPLETE_MARKER"
    if [ "$ALREADY_INITIALIZED" = 1 ]; then
        log "Cache credential updated."
    else
        log "Setup complete."
    fi
}

if [ "${1:-}" = "init" ]; then
    shift
    run_init
    exit 0
fi

# Non-init: locate the initialized directory or fail with a hint. An
# explicit state selection always wins when both candidates exist.
if SELECTED_DIRECTORY="$(state_directory_from_environment)"; then
    if [ -f "$SELECTED_DIRECTORY/.setup-complete" ] \
        && [ -x "$SELECTED_DIRECTORY/nix-static" ]; then
        NIX_DIRECTORY="$SELECTED_DIRECTORY"
    else
        log "The selected ${REVNG_NIX_STATE} nix state is not initialized."
        log "Run: REVNG_NIX_STATE=${REVNG_NIX_STATE} $(basename "$0") init"
        exit 1
    fi
elif [ -f "$CACHE_MARKER" ] && [ -x "$CACHE_BINARY" ]; then
    NIX_DIRECTORY="$CACHE_DIRECTORY"
elif [ -f "$LOCAL_MARKER" ] && [ -x "$LOCAL_BINARY" ]; then
    NIX_DIRECTORY="$LOCAL_DIRECTORY"
else
    log <<EOF
This nix wrapper has not been initialized. Run:

    $(basename "$0") init

to pick a state directory, download the pinned static nix, and
configure the rev.ng caches.
EOF
    exit 1
fi

NIX_BINARY="$NIX_DIRECTORY/nix-static"
CONFIG_FILE="$NIX_DIRECTORY/nix.conf"

# Isolate every per-user path inside NIX_DIRECTORY. NIX_*_HOME are
# honored only when use-xdg-base-directories is on (set via NIX_CONFIG
# below).
export NIX_CACHE_HOME="$NIX_DIRECTORY/nix/cache"
export NIX_STATE_HOME="$NIX_DIRECTORY/nix/state"
export NIX_CONFIG_HOME="$NIX_DIRECTORY/nix/config"
mkdir -p "$NIX_CACHE_HOME" "$NIX_STATE_HOME" "$NIX_CONFIG_HOME"

# NIX_USER_CONF_FILES fully replaces ~/.config/nix/nix.conf:
# substituters, trusted keys, netrc and experimental-features all come
# from our generated nix.conf. use-xdg-base-directories switches on
# the NIX_*_HOME lookup, and the store setting keeps the store itself
# inside NIX_DIRECTORY. Nix enters auto-chroot mode when /nix/store is
# not writable: it creates a user + mount namespace, bind-mounts
# NIX_DIRECTORY/store as /nix/store inside the namespace, and reruns
# itself there.
export NIX_USER_CONF_FILES="$CONFIG_FILE"
export NIX_CONFIG="use-xdg-base-directories = true
store = $NIX_DIRECTORY/store"

# The static nix has no built-in CA bundle; point it at the host's.
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    export NIX_SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
elif [ -f /etc/ssl/cert.pem ]; then
    export NIX_SSL_CERT_FILE="/etc/ssl/cert.pem"
fi

exec -a nix "$NIX_BINARY" "$@"
