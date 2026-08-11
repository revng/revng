{ pkgs, stdenv }:
let
  # The tarball isn't redistributable, so the user has to populate it
  # in the store themselves. To add it run, from the host:
  #
  #   nix-store --add-fixed sha256 ~/Xcode_9.xip-MacOSX.sdk.tar.gz
  sdkTarball = pkgs.requireFile {
    name = "Xcode_9.xip-MacOSX.sdk.tar.gz";
    sha256 = "8aa2fbfd007837fd053b34549082e7681428d9fd896e1383dc7995750275a38a";
    message = ''
      The macOS SDK tarball Xcode_9.xip-MacOSX.sdk.tar.gz is needed.
      Run, from the host shell:

        nix-store --add-fixed sha256 ~/Xcode_9.xip-MacOSX.sdk.tar.gz
    '';
  };
in
stdenv.mkDerivation {
  name = "macos-sdk";
  __structuredAttrs = true;
  useCache = "private";
  src = sdkTarball;
  unpackPhase = "true";
  installPhase = ''
    mkdir -p $out
    tar -xzf ${sdkTarball} -C $out
  '';
}
