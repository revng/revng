{ pkgs, stdenv, python }:
# Tiny derivation that owns just `share/revng/test/` (configuration
# YAMLs + .c/.S/.filecheck/.model.yml/.cfg.yml/... fixtures).
#
# The bulk of revng never reads those files at runtime, but they
# matter for every test/revng iteration — and pre-carve-out, *any*
# edit under share/revng/test invalidated revng's src hash and
# forced a full ~10 min rebuild. This derivation pays only the
# cost of a directory copy, leaving revng's store path stable.
stdenv.mkDerivation {
  name = "revng-test-assets";

  src = pkgs.lib.cleanSourceWith {
    src = ../..;
    filter =
      path: type:
      let
        prefix = toString ../..;
        rel = pkgs.lib.removePrefix (prefix + "/") (toString path);
      in
      rel == "share"
      || rel == "share/revng"
      || pkgs.lib.hasPrefix "share/revng/test" rel;
  };

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/share/revng"
    cp -a share/revng/test "$out/share/revng/test"

    # `for-model-migration` isn't emitted by any revng-qa YAML so
    # `test-configure` fails on the unmatched target type. Skip the
    # YAML so `test/revng` proceeds with what revng-qa provides.
    rm -f "$out/share/revng/test/configuration/revng/model-migration.yml"

    # Some fixture scripts (`tests/prss/websocket_client.py`,
    # `tests/scripting.py`, …) are invoked directly as executables
    # from test rules in `test/revng-prss` and `test/revng-*`. They
    # ship with `#!/usr/bin/env python3` shebangs — `/usr/bin/env`
    # isn't on PATH inside the nix sandbox, so the kernel reports
    # `cannot execute: required file not found`. The test/revng
    # derivation patches its own emitted scripts in `preInstall`;
    # these assets live outside that tree so we patch them here.
    for _f in $(find "$out/share/revng/test" -type f -perm -u+x); do
      sed -i "1{
        s|^#!/usr/bin/env python3.*|#!${python}/bin/python3|
        s|^#!/usr/bin/env bash.*|#!${pkgs.bash}/bin/bash|
        s|^#!/bin/bash.*|#!${pkgs.bash}/bin/bash|
      }" "$_f"
    done
    unset _f

    runHook postInstall
  '';
}
