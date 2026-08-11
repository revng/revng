{
  lib,
  makeWrapper,
  nix,
  ninja,
  openssh,
  python3,
  rsync,
  runCommand,
  zstd,
}:
let
  python = python3.withPackages (
    pythonPackages: with pythonPackages; [
      pynacl
      pyyaml
    ]
  );
  runtimeDependencies = [
    nix
    ninja
    openssh
    rsync
    zstd
  ];
in
runCommand "nix-cache-push"
  {
    nativeBuildInputs = [ makeWrapper ];
    meta.mainProgram = "nix-cache-push";
  }
  ''
    install -Dm755 ${./push.py} "$out/bin/nix-cache-push"
    substituteInPlace "$out/bin/nix-cache-push" \
      --replace-fail '#!/usr/bin/env python3' '#!${python}/bin/python3'
    wrapProgram "$out/bin/nix-cache-push" \
      --prefix PATH : ${lib.makeBinPath runtimeDependencies}
  ''
