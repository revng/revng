{ pkgs }:
# Use a fake npm project to specify JavaScript dependencies
pkgs.stdenv.mkDerivation (finalAttrs: {
  nativeBuildInputs = [
    pkgs.nodejs
    pkgs.pnpm.configHook
  ];
  pname = "revng";
  version = "1.0";
  src = pkgs.lib.cleanSourceWith {
    name = "revng-js-dependencies";
    src = ./.;
    filter = path: type: baseNameOf path != "default.nix";
  };
  installPhase = ''
    pwd
    mkdir -p $out/node_modules
    cp -Tar /build/revng-js-dependencies/node_modules $out/node_modules
  '';
  pnpmDeps = pkgs.pnpm.fetchDeps {
    inherit (finalAttrs) pname version src;
    fetcherVersion = 2;
    hash = "sha256-VxFmVePLXkuBR1kaLj+djwdMqn2uh+m5YM0mSIfXOlo=";
  };
})
