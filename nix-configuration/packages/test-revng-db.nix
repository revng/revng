{ pkgs, stdenv, python,
  revng, revngPythonDependencies, revngPackages,
}:
# Mirrors orchestra's `test/revng-db` (revng-test.yml:76-83):
#   build_dependencies: [ ninja, revng-qa, revng, model-db ]
#   configure_args: --target-type 'revng-db\..*'
#
# The db tests live in revng's share/revng/test/configuration/
# revng-db/*.yml. db-invariants.yml does `sqlite3
# $SOURCES_ROOT/share/revng/prototypes.sqlite < $INPUT | FileCheck`,
# import-prototypes-from-db.yml does `revng analyze
# import-prototypes-from-db --model $INPUT | revng model opt -verify
# | FileCheck $SOURCE`.
#
# Each component lives in its own store path, so we pass them all to
# test-configure as repeated --input-path roots; revng's
# ResourceFinder learns the same list via REVNG_RESOURCES.
let
  revngPythonEnv = revngPythonDependencies.overrideAttrs (old: {
    postInstall = (old.postInstall or "") + ''
      cp -a ${revng}/${python.sitePackages}/. \
        $out/${python.sitePackages}/
    '';
  });

  # Search list used both as --input-path roots for test-configure
  # and as REVNG_RESOURCES for the running revng. Order matters: the
  # first hit wins. revng-test-assets owns the revng-db YMLs (split
  # out of revng so test-fixture edits don't trigger a full revng
  # rebuild); revng-qa owns the bulk of fixture sources.
  searchRoots = [
    revngPackages.revng-qa
    revngPackages.revng-test-assets
    revng
    revngPackages.model-db
  ];
in
stdenv.mkDerivation {
  name = "test/revng-db";

  __structuredAttrs = true;
  unsafeDiscardReferences.out = true;

  unpackPhase = "true";

  nativeBuildInputs = (with pkgs; [
    sqlite      # provides `sqlite3` for db-invariants.yml
    llvm_21     # provides `FileCheck`
  ]) ++ [
    revngPackages.ninjaShellRule
    revng
    revngPythonEnv
  ];

  buildPhase = ''
    echo
  '';

  preInstall = ''
    python3 \
      ${revngPackages.revng-qa}/libexec/revng/test-configure \
      "${revngPackages.revng-test-assets}/share/revng/test/configuration/revng-db/"*.yml \
      --install-path "$PWD" \
      ${pkgs.lib.concatMapStringsSep " " (p: ''--input-path "${p}"'') searchRoots} \
      --destination . \
      --target-type 'revng-db\..*'
    # test-configure emits scripts shebanged with `/usr/bin/env <X>`
    # that don't resolve in the sandbox; rewrite to absolute paths.
    # See test-revng.nix for why this isn't `patchShebangs`.
    for _f in $(find . -maxdepth 2 -type f \( -name "*.py" -o -name "*.sh" -o -name "*.js" \)); do
      sed -i "1{
        s|^#!/usr/bin/env python3.*|#!$(command -v python3)|
        s|^#!/usr/bin/env bash.*|#!$(command -v bash)|
        s|^#!/usr/bin/env node.*|#!$(command -v node)|
        s|^#!/bin/bash.*|#!$(command -v bash)|
      }" "$_f"
    done
    unset _f

    export REVNG_OPTIONS="--debug-log=verify"
    export XDG_CACHE_HOME="$PWD/.cache"
    mkdir -p "$XDG_CACHE_HOME/.cache"

    # Same search list test-configure was given. Each component
    # (revng, revng-qa, model-db) lives in its own store path;
    # ResourceFinder probes them in order.
    export REVNG_RESOURCES="${pkgs.lib.concatMapStringsSep ":" toString searchRoots}"
  '';

  installPhase = ''
    mkdir -p $out
    runHook preInstall

    mkdir -p "$out/log"
    ninja -v -k0 all 2>&1 | tee "$out/log/ninja.log" || true

    grep -oE '^FAILED: [^ ]+' "$out/log/ninja.log" \
      > "$out/log/failed-targets.txt" || true
    echo "test/revng-db: $(wc -l < $out/log/failed-targets.txt) failing target(s); see $out/log/"
  '';

  # Same `repro-setup` / `repro-run` shell helpers test-revng.nix
  # exposes, so `nix develop .#"test/revng-db"` works the same way.
  shellHook = ''
    repro-setup() {
      local wd=''${1:-/tmp/revng-db-test-repro}
      mkdir -p "$wd" || return $?
      (
        cd "$wd" &&
        rm -f build.ninja build.ninja.diag &&
        eval "$preInstall" &&
        echo "Setup complete in $wd."
      )
    }
    repro-run() {
      local target=''${1:?'usage: repro-run TARGET [WORKDIR]'}
      local wd=''${2:-/tmp/revng-db-test-repro}
      (
        cd "$wd" || return $?
        if [[ ! -f build.ninja ]]; then eval "$preInstall"; fi
        if [[ ! -f build.ninja.diag ]] || [[ build.ninja -nt build.ninja.diag ]]; then
          sed -e "s# 2>/dev/null##g" -e "s# || true##g" build.ninja > build.ninja.diag
        fi
        ninja -f build.ninja.diag -v -k0 "$target"
      )
    }
    if [[ -t 1 && -z ''${REPRO_QUIET:-} ]]; then
      echo "[test/revng-db] helpers available: repro-setup, repro-run"
    fi
  '';
}
