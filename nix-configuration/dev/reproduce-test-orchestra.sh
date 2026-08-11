#!/usr/bin/env bash
# Orchestra-side counterpart of nix/dev/reproduce-test.sh.
#
# Sets up a `test/revng`-style ninja build under orchestra (without
# going through `orc install`, so it works even when the broader
# `test/revng` build has unrelated failures) and runs individual
# targets against orchestra's installed `revng` / `revng2`.
#
# Useful for confirming that a failure observed under the nix port
# also reproduces under orchestra — i.e. is upstream-real rather than
# nix-port-induced.
#
# Prerequisites:
#   - $ORCHESTRA_DOTDIR points at /home/nix/orchestra/.orchestra (or
#     the script is run from somewhere `orc` can locate the dotdir on
#     its own).
#   - `orc install revng` has succeeded at least once, so
#     $ORCHESTRA_ROOT/bin/revng and $ORCHESTRA_ROOT/lib*/librevng*
#     exist.
#   - `orc install test/revng-qa` has succeeded — the per-arch test
#     fixtures (calc-mips-…, floating-point-x86-64-…) need to be
#     installed under $ORCHESTRA_ROOT/share/revng/test/tests/.
#
# Usage:
#   nix/dev/reproduce-test-orchestra.sh setup [WORKDIR]
#   nix/dev/reproduce-test-orchestra.sh run   TARGET [WORKDIR]
#   nix/dev/reproduce-test-orchestra.sh shell [WORKDIR]
#
# Defaults: WORKDIR=/tmp/revng-test-repro-orch.
set -euo pipefail

ORCHESTRA_DIR=${ORCHESTRA_DIR:-/home/nix/orchestra}
WD_DEFAULT=/tmp/revng-test-repro-orch

if [[ ! -x "$ORCHESTRA_DIR/venv/bin/orc" ]]; then
    echo "reproduce-test-orchestra: orc not found under $ORCHESTRA_DIR/venv/bin" >&2
    echo "  set ORCHESTRA_DIR=/path/to/orchestra or `orc install revng` first." >&2
    exit 1
fi

setup() {
    local wd=${1:-$WD_DEFAULT}
    mkdir -p "$wd"
    wd=$(cd "$wd" && pwd -P)

    # Run inside `orc shell --component revng` so $ORCHESTRA_ROOT,
    # $PATH, and the rpath for revng's tools are correctly set.
    # `orc` locates `.orchestra` by walking up from CWD, so cd into
    # $ORCHESTRA_DIR before invoking it.
    #
    # for-import-idb.yml and for-import-pe.yml depend on idat64 /
    # MSVC outputs that aren't available without the proprietary
    # components installed under orchestra. Drop those two — every
    # other revng test compiles, which is what we need for parity
    # comparisons against the nix port.
    # orc install revng doesn't currently land share/revng/test/
    # configuration/revng/ under $ORCHESTRA_ROOT (the test-suite YAMLs
    # only live in $BUILD_DIR's share/). Read the revng/*.yml configs
    # straight from the source checkout so this script doesn't depend
    # on a full `orc install test/revng` succeeding.
    ( cd "$ORCHESTRA_DIR" &&
      "$ORCHESTRA_DIR/venv/bin/orc" shell --component revng bash -euo pipefail -c '
        cd '"$wd"'
        rm -rf -- ./*
        REVNG_CFG="$ORCHESTRA_DOTDIR/../sources/revng/share/revng/test/configuration/revng"
        if [[ ! -d "$REVNG_CFG" ]]; then
            echo "expected revng test configs at $REVNG_CFG" >&2
            exit 1
        fi
        ymls=()
        for f in "$ORCHESTRA_ROOT/share/revng/test/configuration/revng-qa"/*.yml; do
            ymls+=("$f")
        done
        for f in "$REVNG_CFG"/*.yml; do
            case "$(basename "$f")" in
                for-import-idb.yml|for-import-pe.yml) continue ;;
            esac
            ymls+=("$f")
        done
        "$ORCHESTRA_ROOT/libexec/revng/test-configure" \
            "${ymls[@]}" \
            --install-path "$ORCHESTRA_ROOT" \
            --destination . \
            --target-type "revng\..*"
    ' )
    echo "Setup complete in $wd."
}

run() {
    local target=${1:?'usage: run TARGET [WORKDIR]'}
    local wd=${2:-$WD_DEFAULT}
    wd=$(cd "$wd" && pwd -P)

    [[ -f "$wd/build.ninja" ]] || setup "$wd" >/dev/null

    # Strip the `2>/dev/null || true` wrappers so SIGABRT / SIGILL
    # backtraces actually reach the terminal — mirrors what
    # nix/dev/reproduce-test.sh does on the nix side.
    if [[ ! -f "$wd/build.ninja.diag" ]] \
        || [[ "$wd/build.ninja" -nt "$wd/build.ninja.diag" ]]; then
        sed -e 's# 2>/dev/null##g' -e 's# || true##g' \
            "$wd/build.ninja" > "$wd/build.ninja.diag"
    fi

    ( cd "$ORCHESTRA_DIR" &&
      "$ORCHESTRA_DIR/venv/bin/orc" shell --component revng bash -euo pipefail -c '
        cd '"$wd"'
        export REVNG_OPTIONS="--debug-log=verify"
        export PYPELINE_STORAGE_PROVIDER="local://?inline"
        ninja -f build.ninja.diag -v -k0 "$@"
    ' bash "$target" )
}

shell() {
    local wd=${1:-$WD_DEFAULT}
    mkdir -p "$wd"
    wd=$(cd "$wd" && pwd -P)

    [[ -f "$wd/build.ninja" ]] || setup "$wd" >/dev/null

    cd "$ORCHESTRA_DIR"
    exec "$ORCHESTRA_DIR/venv/bin/orc" shell --component revng bash --init-file <(cat <<EOF
source /etc/bashrc 2>/dev/null || true
cd "$wd"
export REVNG_OPTIONS="--debug-log=verify"
export PYPELINE_STORAGE_PROVIDER="local://?inline"
echo "Inside orchestra test/revng repro shell. ninja -f build.ninja.diag <target>"
EOF
)
}

cmd=${1:-help}
case "$cmd" in
    setup) shift; setup "$@" ;;
    run)   shift; run   "$@" ;;
    shell) shift; shell "$@" ;;
    *)
        sed -n '/^# Usage:/,/^set -euo/p' "$0" | sed '$d'
        exit 1 ;;
esac
