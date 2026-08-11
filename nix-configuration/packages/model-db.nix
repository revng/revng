{ pkgs, stdenv, revng, revngPackages }:
# model-db: aggregate all available *.yml models into
# share/revng/prototypes.sqlite via `revng model export sqlite`.
# Currently only well-known-models is consumed; rootfs/* and
# win32metadata/pdbs/* models can be added once those layers
# land — model-db will pick them up automatically.
let
  linuxRoots = [
    "ubuntu-20-04-x86-64"
    "ubuntu-22-04-x86-64"
    "ubuntu-24-04-x86-64"
    "ubuntu-24-04-i386"
    "ubuntu-24-04-arm"
    "ubuntu-24-04-aarch64"
    "ubuntu-24-04-s390x"
    "debian-bookworm-mipsel"
    "debian-buster-mips"
  ];
  pdbArchs = [
    "x86-64"
    "i386"
    "aarch64"
  ];
  linuxModels = map (n: {
    name = n;
    drv = revngPackages."rootfs/${n}/models";
  }) linuxRoots;
  pdbModels = map (a: {
    arch = a;
    drv = revngPackages."win32metadata/pdbs/${a}/models";
  }) pdbArchs;
in
stdenv.mkDerivation {
  name = "model-db";
  unpackPhase = "true";
  nativeBuildInputs =
    [
      revng
      revngPackages."test/revng-qa/models"
    ]
    ++ (map (e: e.drv) linuxModels)
    ++ (map (e: e.drv) pdbModels);
  installPhase = ''
    DB_NAME=prototypes.sqlite
    rm -f "$DB_NAME"
    export-to-db() {
      local OS="$1" PLATFORM="$2" PREFIX="$3"
      shift 3
      revng model export sqlite \
        --db "$DB_NAME" \
        --platform "$PLATFORM" \
        --operating-system "$OS" \
        --prefix "$PREFIX" \
        "$@"
    }

    ${pkgs.lib.concatMapStringsSep "\n" (e: ''
      DIR="${e.drv}/share/roots/linux/${e.name}"
      MODELS=$(find "$DIR" -name '*.yml' 2>/dev/null)
      if [ -n "$MODELS" ]; then
        echo "Exporting models from ${e.name} to DB" >&2
        export-to-db Linux "${e.name}" "$DIR" $MODELS
      fi
    '') linuxModels}

    ${pkgs.lib.concatMapStringsSep "\n" (e: ''
      DIR="${e.drv}/share/win32metadata/pdbs/${e.arch}"
      MODELS=$(find "$DIR" -name '*.yml' 2>/dev/null)
      if [ -n "$MODELS" ]; then
        echo "Exporting PDB models for ${e.arch} to DB" >&2
        export-to-db Windows "windows-${e.arch}" "$DIR" $MODELS
      fi
    '') pdbModels}

    WK="${revngPackages."test/revng-qa/models"}/share/revng/test/tests/well-known-models"
    if [ -d "$WK" ]; then
      for MODEL in "$WK"/*.yml; do
        [ -f "$MODEL" ] || continue
        BASENAME="$(basename "$MODEL" .yml)"
        PLATFORM="linux-$(echo "$BASENAME" | grep -oP 'libc-\K[^-]+' || echo unknown)"
        echo "Exporting well-known model $BASENAME to DB" >&2
        export-to-db Linux "$PLATFORM" "$WK" "$MODEL"
      done
    fi

    mkdir -p "$out/share/revng"
    cp "$DB_NAME" "$out/share/revng/$DB_NAME"
  '';
}
