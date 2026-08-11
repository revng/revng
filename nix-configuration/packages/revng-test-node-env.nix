{ pkgs, revng, revngJavascriptDependencies }:
# Node interpreter pre-loaded with every JS dep that test/revng needs
# at runtime — `s3rver` (for revng.test-s3-storage), the `revng-model`
# package (for revng.typescript-model-serialization-roundtrip's
# `require('revng-model')`), and the typescript / npm tooling the
# `revng test-docs` doctests invoke.
#
# Conceptually analogous to `python.withPackages` on the Python side:
# the inner `nodejs` binary is wrapped so `require()` and CLI tools
# already see the right `node_modules` without callers having to
# export NODE_PATH or assemble a search path themselves.
let
  inherit (pkgs) lib;
  # Single `node_modules` tree covering both groups: `revng-model` /
  # `revng-pipeline-description` come from revng (CMake's
  # typescript/CMakeLists.txt installs them under
  # `lib/node_modules/<name>`); everything else (`s3rver`,
  # `typescript`, `eslint`, …) comes from revngJavascriptDependencies.
  combinedNodeModules = pkgs.runCommand "revng-test-node-modules" { } ''
    mkdir -p "$out"
    # Preserve symlinks (no `-L`). revngJavascriptDependencies uses
    # pnpm's `.pnpm/<pkg@ver>/node_modules/<pkg>` layout with the
    # top-level `node_modules/<pkg>` entries pointing into it; if
    # we dereference, runtime `require()` traversal of transitive
    # deps breaks (s3rver -> http-errors -> statuses, …).
    cp -aT ${revngJavascriptDependencies}/node_modules "$out"
    chmod -R u+w "$out"
    for pkg in ${revng}/lib/node_modules/*; do
      [ -e "$pkg" ] || continue
      name="$(basename "$pkg")"
      rm -rf "$out/$name"
      # revng-model is a single self-contained package directory
      # — no internal symlinks, so the deref is fine here.
      cp -aLT "$pkg" "$out/$name"
    done
    chmod -R u+w "$out"
  '';

  # Names of executables under node_modules/.bin that we want
  # available on PATH (each gets a wrapper that bakes the NODE_PATH
  # in, so callers don't need to export it).
  wrappedBins = [
    "node"
    "npm"
    "npx"
    "tsc"
    "s3rver"
  ];
in
pkgs.stdenvNoCC.mkDerivation {
  name = "revng-test-node-env";

  dontUnpack = true;
  dontConfigure = true;
  dontBuild = true;

  nativeBuildInputs = [ pkgs.makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p "$out/lib" "$out/bin"
    ln -sn ${combinedNodeModules} "$out/lib/node_modules"

    # node itself isn't in node_modules/.bin — wrap the upstream
    # nodejs binary directly so `require('revng-model')` resolves
    # without callers fiddling with NODE_PATH.
    makeWrapper "${pkgs.nodejs}/bin/node" "$out/bin/node" \
      --set NODE_PATH "$out/lib/node_modules"

    # Everything else lives in node_modules/.bin; wrap each.
    for name in ${lib.escapeShellArgs (builtins.filter (n: n != "node") wrappedBins)}; do
      src="${combinedNodeModules}/.bin/$name"
      if [ ! -e "$src" ]; then
        echo "revng-test-node-env: missing $src" >&2
        exit 1
      fi
      makeWrapper "$src" "$out/bin/$name" \
        --prefix PATH : "$out/bin" \
        --set NODE_PATH "$out/lib/node_modules"
    done

    runHook postInstall
  '';
}
