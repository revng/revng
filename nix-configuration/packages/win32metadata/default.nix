{ pkgs, stdenv, python, llvm, msvc, revng, mkModels }:
let
  # Microsoft's win32metadata: the .winmd files we'll turn into PDBs
  # later.
  win32metadata = pkgs.fetchFromGitHub {
    owner = "microsoft";
    repo = "win32metadata";
    rev = "223f4b9723d8fb7c83c286b6b4ad75dff18985c4";
    hash = "sha256-4FamAMIy60d4gUbejX7O6TEyWwUevUtVMOC35e19zbk=";
  };

  # mkPdbs: drives compile-to-pdb.py to produce a directory of
  # *.pdb files from win32metadata for one target architecture.
  # Uses the patched clang from our own `llvm` derivation plus
  # the VC headers from the matching vc19 toolchain.
  mkPdbs =
    {
      name,
      targetTriple,
      vcTriple,
      archRspFlags,
    }:
    let
      vcToolchain = msvc.byTriple.${vcTriple};
    in
    stdenv.mkDerivation {
      name = "win32metadata-pdbs-${name}";
      __structuredAttrs = true;
      # PDBs are compiled against the vc19 VC/SDK headers, so their
      # $out embeds Microsoft-copyrighted material.
      useCache = "private";
      unpackPhase = "true";
      nativeBuildInputs = [
        llvm
        pkgs.ninja
        python
        vcToolchain
      ];
      buildPhase = ''
        mkdir build
        python3 ${./compile-to-pdb.py} \
          --win32meta-root ${win32metadata} \
          --clang ${llvm}/libexec/clang \
          --lld-link ${llvm}/libexec/lld-link \
          --vc19-include ${vcToolchain}/lib/vc/${vcTriple}/VC/include \
          --target-triple "${targetTriple}" \
          ${archRspFlags} \
          --output-dir build
        cd build
        # `-j$(nproc)` OOMs on hosts with <~4 GB/core: each parallel
        # clang invocation peaks around 2 GB. Honor nix's `--cores N`
        # (exported as $NIX_BUILD_CORES) so callers can cap it.
        ninja -j"''${NIX_BUILD_CORES:-$(nproc)}"
      '';
      installPhase = ''
        # buildPhase ended inside build/; we're already there.
        mkdir -p "$out/share/win32metadata/pdbs/${name}"
        for PDB in *.pdb; do
          [ -f "$PDB" ] && cp "$PDB" "$out/share/win32metadata/pdbs/${name}/"
        done
      '';
    };

  pdbs = {
    "win32metadata/pdbs/x86-64" = mkPdbs {
      name = "x86-64";
      targetTriple = "x86_64-pc-windows-msvc";
      vcTriple = "x86_64-winsdk-vc19";
      archRspFlags = "--arch-rsp baseSettings.x64.rsp";
    };
    "win32metadata/pdbs/i386" = mkPdbs {
      name = "i386";
      targetTriple = "i386-pc-windows-msvc";
      vcTriple = "i386-winsdk-vc19";
      archRspFlags = "--arch-rsp baseSettings.x86.rsp --arch-rsp baseSettings.32.rsp";
    };
    "win32metadata/pdbs/aarch64" = mkPdbs {
      name = "aarch64";
      targetTriple = "aarch64-pc-windows-msvc";
      vcTriple = "aarch64-winsdk-vc19";
      archRspFlags = "--arch-rsp baseSettings.arm64.rsp --arch-rsp baseSettings.64.rsp";
    };
  };

  pdbsModels = {
    "win32metadata/pdbs/x86-64/models" = mkModels {
      name = "win32metadata-pdbs-x86-64-models";
      revngBin = revng;
      buildInputs = [ pdbs."win32metadata/pdbs/x86-64" ];
      installDest = "share/win32metadata/pdbs/x86-64";
      importCommand = "revng model import debug-info";
      findInputs = ''
        PDB_DIR="${pdbs."win32metadata/pdbs/x86-64"}/share/win32metadata/pdbs/x86-64"
        for PDB in "$PDB_DIR"/*.pdb; do
          [ -f "$PDB" ] || continue
          REL="$(basename "$PDB")"
          OUTPUT="$OUTPUT_DIR/$REL.yml"
          mkdir -p "$(dirname "$OUTPUT")"
          echo "build $OUTPUT: import $PDB" >> build.ninja
        done
      '';
    };
    "win32metadata/pdbs/i386/models" = mkModels {
      name = "win32metadata-pdbs-i386-models";
      revngBin = revng;
      buildInputs = [ pdbs."win32metadata/pdbs/i386" ];
      installDest = "share/win32metadata/pdbs/i386";
      importCommand = "revng model import debug-info";
      findInputs = ''
        PDB_DIR="${pdbs."win32metadata/pdbs/i386"}/share/win32metadata/pdbs/i386"
        for PDB in "$PDB_DIR"/*.pdb; do
          [ -f "$PDB" ] || continue
          REL="$(basename "$PDB")"
          OUTPUT="$OUTPUT_DIR/$REL.yml"
          mkdir -p "$(dirname "$OUTPUT")"
          echo "build $OUTPUT: import $PDB" >> build.ninja
        done
      '';
    };
    "win32metadata/pdbs/aarch64/models" = mkModels {
      name = "win32metadata-pdbs-aarch64-models";
      revngBin = revng;
      buildInputs = [ pdbs."win32metadata/pdbs/aarch64" ];
      installDest = "share/win32metadata/pdbs/aarch64";
      importCommand = "revng model import debug-info";
      findInputs = ''
        PDB_DIR="${pdbs."win32metadata/pdbs/aarch64"}/share/win32metadata/pdbs/aarch64"
        for PDB in "$PDB_DIR"/*.pdb; do
          [ -f "$PDB" ] || continue
          REL="$(basename "$PDB")"
          OUTPUT="$OUTPUT_DIR/$REL.yml"
          mkdir -p "$(dirname "$OUTPUT")"
          echo "build $OUTPUT: import $PDB" >> build.ninja
        done
      '';
    };
  };
in
{
  inherit win32metadata;
} // pdbs // pdbsModels
