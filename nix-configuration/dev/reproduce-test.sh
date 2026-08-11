#!/usr/bin/env bash
# Thin wrapper around `nix develop .#"test/revng"`'s `repro-setup` /
# `repro-run` shell functions. Use this for one-off invocations from
# outside the dev shell; if you're iterating, prefer running the
# functions directly inside `nix develop .#"test/revng"` so you don't
# pay the ~10s nix-evaluation cost on each `run`.
#
# Usage:
#   nix/dev/reproduce-test.sh setup [WORKDIR]
#   nix/dev/reproduce-test.sh run TARGET [WORKDIR]
#   nix/dev/reproduce-test.sh shell [WORKDIR]
set -euo pipefail

cmd=${1:-help}
case "$cmd" in
    setup) shift; argv=("repro-setup"  "$@") ;;
    run)   shift; argv=("repro-run"    "$@") ;;
    shell)
        # Drop straight into a nix develop shell, cd'd into the
        # workdir, with `eval "$preInstall"` already done.
        wd=${2:-/tmp/revng-test-repro}
        mkdir -p "$wd"
        wd=$(cd "$wd" && pwd -P)
        exec nix develop "$(git rev-parse --show-toplevel)#\"test/revng\"" \
            --command bash --init-file <(cat <<EOF
source /etc/bashrc 2>/dev/null || true
cd "$wd"
repro-setup "$wd" >/dev/null
EOF
)
        ;;
    *)
        sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed '$d'
        exit 1 ;;
esac

# nix develop --command bash -c '…' inherits the dev shell's env vars
# but not shellHook (that one is wired into the interactive entry).
# Eval it ourselves so `repro-setup` / `repro-run` are defined.
exec env REPRO_QUIET=1 nix develop "$(git rev-parse --show-toplevel)#\"test/revng\"" \
    --command bash -c 'eval "$shellHook"; "$@"' bash "${argv[@]}"
