{ mkModels, revng, revngPackages }:
# well-known-models: import each compiled-with-debug-info
# binary shipped by test/revng-qa into a per-binary .yml model.
mkModels {
  name = "test-revng-qa-models";
  revngBin = revng;
  buildInputs = [ revngPackages."test/revng-qa" ];
  installDest = "share/revng/test/tests/well-known-models";
  findInputs = ''
    WELL_KNOWN_DIR="${revngPackages."test/revng-qa"}/share/revng/test/tests/well-known-models"
    for BINARY in "$WELL_KNOWN_DIR/"*revng-qa.compiled-with-debug-info-*; do
      case "$BINARY" in *.yml) continue ;; esac
      BASENAME="$(basename "$BINARY")"
      OUTPUT="$OUTPUT_DIR/''${BASENAME}.yml"
      mkdir -p "$(dirname "$OUTPUT")"
      echo "build $OUTPUT: import $BINARY" >> build.ninja
    done
  '';
  importCommand = "REVNG_NO_FETCH_DEBUG_INFO=1 revng analyze import-binary";
}
