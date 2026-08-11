{ pkgs, stdenv, python,
  revng, revngPythonDependencies, revngPackages,
}:
stdenv.mkDerivation {
  name = "test/revng-prss";

  __structuredAttrs = true;
  unsafeDiscardReferences.out = true;

  unpackPhase = "true";

  nativeBuildInputs =
    (with pkgs; [
      jq
      ninja
      # tests/prss/common starts a local postgres + spawns
      # revng2's rss-server; we need both binaries on PATH
      # plus curl for the wait_for_status loop.
      postgresql
      curl
    ])
    ++ [
      revng
      revngPackages."test/revng-qa"
      revngPythonDependencies
      (python.withPackages (
        ps: with ps; [
          jinja2
          pyyaml
        ]
      ))
    ];

  buildPhase = ''
    echo
  '';

  installPhase = ''
    mkdir -p "$out"

    python3 \
      ${revngPackages.revng-qa}/libexec/revng/test-configure \
      "${revngPackages.revng-qa}/share/revng/test/configuration/revng-qa/"*.yml \
      "${revngPackages.revng-test-assets}/share/revng/test/configuration/revng-prss/"*.yml \
      --install-path "$PWD" \
      --input-path "${revngPackages.revng-qa}" \
      --input-path "${revngPackages."test/revng-qa"}" \
      --input-path "${revngPackages.revng-test-assets}" \
      --input-path "${revng}" \
      --destination . \
      --target-type 'revng-prss\..*'

    export REVNG_RESOURCES="${revngPackages.revng-qa}:${revngPackages."test/revng-qa"}:${revngPackages.revng-test-assets}:${revng}"

    # WIP: same patchShebangs / PYTHONPATH / PYPELINE_STORAGE_
    # PROVIDER / `shell` strip / timeout / sh symlink dance as
    # test/revng. See those comments for rationale.
    patchShebangs --build .

    export PYTHONPATH="${revng}/${python.sitePackages}:${revngPythonDependencies}/${python.sitePackages}''${PYTHONPATH:+:$PYTHONPATH}"
    export PYPELINE_STORAGE_PROVIDER="local://?inline"

    grep -v 'shell =' build.ninja > build2.ninja
    mv build2.ninja build.ninja

    sed -i \
      -e 's| revng2 | timeout 600 revng2 |g' \
      -e 's| revng artifact| timeout 600 revng artifact|g' \
      build.ninja

    ln -s "$(command -v bash)" sh
    export XDG_CACHE_HOME="$PWD/.cache"
    mkdir -p "$XDG_CACHE_HOME/.cache"

    mkdir -p "$out/log"
    ninja -v -k0 all 2>&1 | tee "$out/log/ninja.log" || true

    grep -oE 'FAILED: \[code=[0-9]+\] [^ ]+' "$out/log/ninja.log" \
      > "$out/log/failed-targets.txt" || true
    echo "test/revng-prss: $(wc -l < $out/log/failed-targets.txt) failing target(s); see $out/log/"
  '';

}
