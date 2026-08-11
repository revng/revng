{ pkgs, stdenv, inputs }:
let
  aws-crt-cpp = (
    pkgs.callPackage "${inputs.nixpkgs}/pkgs/by-name/aw/aws-crt-cpp/package.nix" {
      stdenv = stdenv;
    }
  );

  aws-sdk-cpp =
    (pkgs.callPackage "${inputs.nixpkgs}/pkgs/by-name/aw/aws-sdk-cpp/package.nix" {
      stdenv = stdenv;

      aws-crt-cpp = aws-crt-cpp;

      # Only build the APIs we're interested in
      apis = [ "s3" ];
    }).overrideAttrs
      (oldAttrs: {
        cmakeFlags = oldAttrs.cmakeFlags ++ [
          "-DENABLE_TESTING=OFF"
          "-DFORCE_CURL=ON"
          "-DENABLE_UNITY_BUILD=OFF"
          "-DENABLE_RTTI=OFF"
          "-DCPP_STANDARD=20"
        ];
      });
in
{
  inherit aws-crt-cpp aws-sdk-cpp;
}
