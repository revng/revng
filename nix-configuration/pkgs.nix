{ inputs, system }:
let
  ccacheOverlay = (
    self: super: {
      ccacheWrapper = super.ccacheWrapper.override {
        extraConfig = ''
          export CCACHE_COMPRESS=1
          export CCACHE_SLOPPINESS=random_seed
          export CCACHE_DIR="/nix/var/cache/ccache"
          export CCACHE_UMASK=007
          if [ ! -d "$CCACHE_DIR" ]; then
            echo "====="
            echo "Directory '$CCACHE_DIR' does not exist"
            echo "Please create it with:"
            echo "  sudo mkdir -m0770 '$CCACHE_DIR'"
            echo "  sudo chown root:nixbld '$CCACHE_DIR'"
            echo "====="
            exit 1
          fi
          if [ ! -w "$CCACHE_DIR" ]; then
            echo "====="
            echo "Directory '$CCACHE_DIR' is not accessible for user $(whoami)"
            echo "Please verify its access permissions"
            echo "====="
            exit 1
          fi
        '';
      };
    }
  );
  pkgs = import inputs.nixpkgs {
    inherit system;
    # overlays = [ ccacheOverlay ];
  };
  pkgs-2505 = import inputs.nixpkgs-2505 {
    inherit system;
    # overlays = [ ccacheOverlay ];
  };

  # Pin Python to 3.14.x.
  python = pkgs.python314;

  # Adopt:
  #
  # * clang as a compiler
  # * libc++ as C++ standard library
  # * mold as linker
  #
  # Pin to LLVM 16 (matching orchestra/clang-release): revng links its
  # analysis passes against LLVM 16 internals via packages/llvm.nix,
  # so the runtime libc++ must match. With libcxx 21 the decompiler
  # picks the wrong return ABI for scalar-returning functions and
  # several Clift passes (computeBestTraversal et al.) SIGILL.
  # llvmPackages_16 was dropped from our primary nixpkgs as obsolete,
  # so we source it from the pinned pkgs-2505 (NixOS 25.05) instead.
  stdenv = (pkgs.useMoldLinker pkgs-2505.llvmPackages_16.libcxxStdenv);
  ccacheStdenv = stdenv;
  # ccacheStdenv = pkgs.ccacheStdenv.override {
  #   stdenv = stdenv;
  #   extraConfig = ''
  #     export CCACHE_DIR="''${CCACHE_DIR:-/nix/var/cache/ccache}"
  #     export CCACHE_COMPRESS=1
  #     export CCACHE_SLOPPINESS=random_seed
  #     export CCACHE_UMASK=007
  #   '';
  # };
in
{
  inherit pkgs pkgs-2505 python stdenv ccacheStdenv;
}
