{ pkgs, stdenv }:
{
  # Helper used by every `*/models` derivation: walks a directory
  # tree, runs `revng analyze import-binary` (or `revng model
  # import debug-info` for PDBs) against every input file via a
  # generated build.ninja, and installs the resulting `*.yml`
  # files under installDest.
  #
  # `revngBin` is the absolute path of the revng package to use.
  # `importCommand` is the import invocation (no trailing $in -o $out).
  # `findInputs` is shell that emits absolute paths of inputs to import.
  # `extraPreNinja` is run before ninja (e.g. to seed a revng cache).
  # `installDest` is the directory under $out where *.yml files land.
  mkModels =
    {
      name,
      revngBin,
      buildInputs ? [ ],
      findInputs,
      importCommand,
      extraPreNinja ? "",
      installDest,
    }:
    stdenv.mkDerivation {
      inherit name;
      __structuredAttrs = true;
      unsafeDiscardReferences.out = true;
      unpackPhase = "true";
      nativeBuildInputs = [
        pkgs.ninja
        revngBin
      ] ++ buildInputs;
      buildPhase = ''
        mkdir -p $BUILD_DIR
        cd $BUILD_DIR
        OUTPUT_DIR="$PWD/models"
        cat > build.ninja <<EOF
        rule import
          command = ${importCommand} \$in -o \$out
          description = Importing \$in
        EOF
        ${findInputs}
      '';
      installPhase = ''
        ${extraPreNinja}
        # WIP: tolerate per-PDB upstream PDBImporterImpl crashes
        # so the rest of the PDB models still land in $out.
        ninja -v -k0 || true
        mkdir -p "$out/${installDest}"
        if [ -d models ]; then
          cd models && find . -name "*.yml" -exec install -Dm644 {} "$out/${installDest}/{}" \;
        fi
      '';
      BUILD_DIR = "build";
    };
}
