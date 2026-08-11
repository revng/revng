{ pkgs, stdenv, python, revng-qa, crossToolchains, ninjaShellRule }:
# Pending: builds the revng-qa IDB test artifacts (idat64-produced
# *.i64 files) from the same well-known-models sources. Requires
# idat64, which we don't ship yet, so the build will fail until an
# IDA toolchain is wired in. Nothing in the flake depends on this
# attr today.
stdenv.mkDerivation {
  name = "revng-qa-idb";

  unpackPhase = "true";

  nativeBuildInputs =
    (with pkgs; [
      binutils
      gcc
    ])
    ++ crossToolchains
    ++ [
      revng-qa
      ninjaShellRule
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
    mkdir -p $out
    python3 \
      ${revng-qa}/libexec/revng/test-configure \
      "${revng-qa}/share/revng/test/configuration/revng-qa-idb/"*.yml \
      --install-path "${revng-qa}" \
      --destination . \
      --target-type 'revng-qa-idb\..*'
    sed -i "s|shell = /bin/bash|shell = ${pkgs.bash}/bin/bash|g" build.ninja
    export XDG_CACHE_HOME="$PWD/.cache"
    mkdir -p "$XDG_CACHE_HOME/.cache"
    PATH="$PWD:$PATH" ninja -v all
    mkdir -p "$out/share"
    cp -a share/revng "$out/share/"
  '';
}
