{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/e8b384078f32ae8bdf7ded43e035b541d61a8cad";

    nixpkgs-2505.url = "https://github.com/NixOS/nixpkgs/archive/refs/heads/nixos-25.05-small.tar.gz";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    revng-qa.url = "github:revng/revng-qa/baa655e348459a15599dea5a640e44fb97423237";
    revng-qa.flake = false;

    nixpkgs-lib.url = "github:nix-community/nixpkgs.lib";
  };

  outputs =
    inputs@{ self, ... }:
    let
      system = "x86_64-linux";
    in
    {
      packages.${system} = import ./nix-configuration { inherit self inputs system; };
    };
}
