{ pkgs }:
# Ninja pinned at v1.11.0 with the shell-for-rule.patch applied so
# that build.ninja rules can carry a `shell = <path>` binding and
# dispatch their command through that shell. revng-qa's
# test-configure relies on this to wrap every rule in a sourced
# common.sh.
pkgs.ninja.overrideAttrs (oldAttrs: rec {
  version = "1.11.0";
  src = pkgs.fetchFromGitHub {
    owner = "ninja-build";
    repo = "ninja";
    rev = "v${version}";
    hash = "sha256-xZwMdwvg29lauHKk9M318Vz7pXZFhf3kFcyOTBdjmJM=";
  };
  # shell-for-rule.patch is rebased on top of nixpkgs's
  # `0001-spawn-sh-instead-of-bin-sh.patch`: when no `shell = X`
  # binding is set we use posix_spawnp("sh") so the sandbox's
  # PATH-resolved sh wins (the upstream patch defaulted to
  # `/bin/sh`, which doesn't exist inside the nix sandbox).
  patches = (oldAttrs.patches or [ ]) ++ [ ./shell-for-rule.patch ];
})
