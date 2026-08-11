{ pkgs, inputs }:
(pkgs.lib.fix (
  self:
  pkgs.callPackage "${inputs.nixpkgs}/pkgs/development/libraries/boost/1.81.nix" {
    stdenv = pkgs.llvmPackages_21.libcxxStdenv;

    # Use the right version of boost-build.
    # This has been copied from nixpkgs.
    boost-build = pkgs.boost-build.override { useBoost = self; };
  }
)).overrideAttrs
  (oldAttrs: {
    # Build only the libraries we're interseted in
    configureFlags = oldAttrs.configureFlags ++ [ "--with-libraries=test" ];
  })
