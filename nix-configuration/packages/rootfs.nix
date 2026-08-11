{ pkgs, stdenv, revng }:
let
  # ---------------------------------------------------------------
  # model-db tree: rootfs/*, win32metadata, win32metadata/pdbs/*,
  # test/revng-qa/models, model-db.
  # ---------------------------------------------------------------

  # Linux rootfs helper. Runs debootstrap --download-only inside a
  # fixed-output derivation (network-allowed), extracts every .deb
  # in place, then trims the result to ELF binaries + symlinks +
  # ld.so.conf. The resulting tree lives under
  # share/roots/linux/<name> where revng / fetch-debuginfo expect
  # to find it.
  mkRootfs =
    {
      name,
      codename,
      architecture,
      url,
      operatingSystem,
      packages_,
      outputHash,
    }:
    let
      components =
        if operatingSystem == "ubuntu" then
          "main,restricted,universe,multiverse"
        else
          "main,contrib,non-free";
    in
    stdenv.mkDerivation {
      name = "rootfs-${name}";
      outputHashAlgo = "sha256";
      outputHashMode = "recursive";
      inherit outputHash;
      unpackPhase = "true";
      # The trimmed rootfs intentionally retains symlinks whose
      # targets were removed (non-ELF binaries got deleted).
      dontCheckForBrokenSymlinks = true;
      nativeBuildInputs = with pkgs; [
        debootstrap
        fakeroot
        dpkg
        cacert
        zstd
        xz
        gzip
      ];
      buildPhase = ''
        export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        fakeroot debootstrap \
          --no-check-gpg \
          --arch="${architecture}" \
          --components="${components}" \
          --include="${packages_}" \
          --download-only \
          "${codename}" \
          rootfs/ \
          "${url}" || true

        # Extract every .deb in place.
        find rootfs -name "*.deb" | while read DEB; do
          mkdir -p temp && cd temp
          ar x "../$DEB"
          cd ../rootfs
          if [ -e "../temp/data.tar"* ]; then
            tar --skip-old-files -xaf "../temp/data.tar"*
          fi
          cd .. && rm -rf temp
        done

        test "$(find rootfs/ -name 'libc.so*' | wc -l)" -ge 1 \
          || { echo "debootstrap ${name} failed: no libc"; exit 1; }

        # Trim non-ELF except for ld.so.conf{,.d/*} and symlinks.
        find rootfs -not -type d | while read F; do
          [ -L "$F" ] && continue
          REL="''${F#rootfs}"
          [ "$REL" = "/etc/ld.so.conf" ] && continue
          case "$REL" in /etc/ld.so.conf.d/*) continue ;; esac
          if head -c 4 "$F" 2>/dev/null | grep -q $'\x7fELF'; then continue; fi
          rm -f "$F"
        done

        chmod -R u+rwX rootfs/

        # Make absolute symlinks relative.
        find rootfs -type l | while read L; do
          T="$(readlink "$L")"
          if [ "''${T#/}" != "$T" ]; then
            D="$(dirname "$L")"
            R="$(realpath -m --relative-to="$D" "rootfs$T")"
            ln -sfn "$R" "$L"
          fi
        done
        find rootfs -type d -empty -delete
      '';
      installPhase = ''
        mkdir -p "$out/share/roots/linux/${name}"
        cp -a rootfs/* "$out/share/roots/linux/${name}/"
        chmod -R u+rwX "$out/share/roots/linux/${name}/"
      '';
    };

  # rootfs/X/debug-info wrapper: runs `revng model fetch-debuginfo`
  # on every ELF in a rootfs and stuffs the resulting symbols cache
  # under share/roots/linux/<name>/symbols-cache.
  mkRootfsDebugInfo =
    {
      name,
      rootfs,
      outputHash,
    }:
    stdenv.mkDerivation {
      name = "rootfs-${name}-debug-info";
      outputHashAlgo = "sha256";
      outputHashMode = "recursive";
      inherit outputHash;
      unpackPhase = "true";
      nativeBuildInputs = [
        revng
        rootfs
        pkgs.ninja
        pkgs.cacert
      ];
      buildPhase = ''
        export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        ROOTFS_DIR="${rootfs}/share/roots/linux/${name}"
        export XDG_CACHE_HOME="$PWD/cache"
        mkdir -p "$XDG_CACHE_HOME" .flags
        cat > build.ninja <<EOF
        rule fetch_debuginfo
          command = revng model fetch-debuginfo \$in || true && touch \$out
          description = fetch-debuginfo \$in
        EOF
        find "$ROOTFS_DIR" -type f ! -path "$ROOTFS_DIR/symbols-cache/*" | \
          while read -r ELF; do
            if head -c 4 "$ELF" 2>/dev/null | grep -q $'\x7fELF'; then
              HASH=$(sha256sum <<< "$ELF" | cut -d' ' -f1)
              ESC=$(sed -e 's| |$ |g' <<< "$ELF")
              echo "build .flags/$HASH: fetch_debuginfo $ESC" >> build.ninja
            fi
          done
        ninja -v
      '';
      installPhase = ''
        SYMS_SRC="$PWD/cache/revng/debug-symbols/elf"
        SYMS_DST="$out/share/roots/linux/${name}/symbols-cache"
        if [ -d "$SYMS_SRC" ]; then
          mkdir -p "$SYMS_DST"
          cp -a "$SYMS_SRC"/* "$SYMS_DST/" || true
        else
          # Always produce an output so downstream paths exist.
          mkdir -p "$SYMS_DST"
        fi
      '';
    };

  # mkRootfsModels: drives `revng analyze import-binary` over every
  # ELF in a rootfs, with the rootfs's symbols-cache pre-seeded into
  # revng's XDG cache and a per-rootfs configuration.yml pointing
  # revng at the rootfs so its DwarfImporter can chase .gnu_debuglink
  # / build-id references back to debug-info files.
  mkRootfsModels =
    {
      name,
      architecture,
      rootfs,
      debugInfo,
    }:
    stdenv.mkDerivation {
      name = "rootfs-${name}-models";
      unpackPhase = "true";
      nativeBuildInputs = [
        revng
        pkgs.ninja
        rootfs
        debugInfo
      ];
      buildPhase = ''
        SOURCE_DIR="${rootfs}/share/roots/linux/${name}"
        export XDG_CACHE_HOME="$PWD/cache"
        mkdir -p "$XDG_CACHE_HOME/revng/debug-symbols/elf"
        if [ -d "${debugInfo}/share/roots/linux/${name}/symbols-cache" ]; then
          cp -a "${debugInfo}/share/roots/linux/${name}/symbols-cache"/* \
            "$XDG_CACHE_HOME/revng/debug-symbols/elf/" || true
        fi

        export XDG_CONFIG_HOME="$PWD/config"
        mkdir -p "$XDG_CONFIG_HOME/revng"
        cat > "$XDG_CONFIG_HOME/revng/configuration.yml" <<EOF
        rootfs:
          ${name}:
            path: $SOURCE_DIR
            architecture: ${architecture}
            operating-system: Linux
        EOF

        OUTPUT_DIR="$PWD/models"
        cat > build.ninja <<EOF
        rule import
          command = REVNG_NO_FETCH_DEBUG_INFO=1 revng analyze import-binary \$in -o \$out
          description = Importing \$in
        EOF

        # Per-rootfs cap: set to a positive integer to limit the
        # number of ELFs imported (useful for cutting build time on
        # rootfs flavours where a few minutes of coverage is enough);
        # 0 (current default) means import every ELF — multiple hours
        # per rootfs, ~5 days for the whole tree.
        MAX_BINARIES=0
        find "$SOURCE_DIR" \
          -not -path "$SOURCE_DIR/symbols-cache/*" \
          -not -path "*/debug/.build-id/*" \
          -not -path "*/lib/debug/*" \
          -not -name "*.debug" \
          -type f > all-files.list
        COUNT=0
        while IFS= read -r ELF; do
          if head -c 4 "$ELF" 2>/dev/null | grep -q $'\x7fELF'; then
            COUNT=$((COUNT + 1))
            if [ "$MAX_BINARIES" -gt 0 ] && [ "$COUNT" -gt "$MAX_BINARIES" ]; then
              break
            fi
            REL="''${ELF#$SOURCE_DIR/}"
            OUT="$OUTPUT_DIR/$REL.yml"
            mkdir -p "$(dirname "$OUT")"
            ESC=$(sed 's| |$ |g' <<< "$ELF")
            echo "build $OUT: import $ESC" >> build.ninja
          fi
        done < all-files.list

        # WIP: -k0 + || true tolerates upstream revng crashes that
        # land per-binary (notably BinaryImporterHelper on Debian
        # MIPS ELFs and PDBImporterImpl::populateTypes for some
        # generated PDBs).
        ninja -v -k0 || true
      '';
      installPhase = ''
        if [ -d models ]; then
          mkdir -p "$out/share/roots/linux/${name}"
          cd models && find . -name "*.yml" -exec install -Dm644 {} "$out/share/roots/linux/${name}/{}" \;
        else
          mkdir -p "$out/share/roots/linux/${name}"
        fi
      '';
    };

  # The 9 supported Linux rootfs configurations. Each is a
  # fixed-output derivation: the outputHash is populated after the
  # first successful build (debootstrap is non-deterministic over
  # time, but a single .deb set hashed once stays valid until the
  # mirror moves).
  rootfses = {
    "rootfs/ubuntu-20-04-x86-64" = mkRootfs {
      name = "ubuntu-20-04-x86-64";
      codename = "focal";
      architecture = "amd64";
      url = "http://archive.ubuntu.com/ubuntu/";
      operatingSystem = "ubuntu";
      packages_ = "libfuse2,libc6-dbg";
      outputHash = "sha256-aVeiMLIDfBrA8cTrdTm/VH9DNfCZt2fDA0E+wzjrVkE=";
    };
    "rootfs/ubuntu-22-04-x86-64" = mkRootfs {
      name = "ubuntu-22-04-x86-64";
      codename = "jammy";
      architecture = "amd64";
      url = "http://archive.ubuntu.com/ubuntu/";
      operatingSystem = "ubuntu";
      packages_ = "libfuse2,libc6-dbg";
      outputHash = "sha256-FODGTaDHNdovdCnTuRnd4Z+mxntUHbgJp5K0LWinFxM=";
    };
    "rootfs/ubuntu-24-04-x86-64" = mkRootfs {
      name = "ubuntu-24-04-x86-64";
      codename = "noble";
      architecture = "amd64";
      url = "http://archive.ubuntu.com/ubuntu/";
      operatingSystem = "ubuntu";
      packages_ = "libfuse3-3,libc6-dbg";
      outputHash = "sha256-cgng+8fusATdWBiaToQvAzb1iwYfM4I4V8Ol8Tnt3xw=";
    };
    "rootfs/ubuntu-24-04-i386" = mkRootfs {
      name = "ubuntu-24-04-i386";
      codename = "noble";
      architecture = "i386";
      url = "http://archive.ubuntu.com/ubuntu/";
      operatingSystem = "ubuntu";
      packages_ = "libfuse3-3,libc6-dbg";
      outputHash = "sha256-HTKCrHIyaOh2RUm8boHw6FE0/QZOO5e1OF2ga5jGk3I=";
    };
    "rootfs/ubuntu-24-04-arm" = mkRootfs {
      name = "ubuntu-24-04-arm";
      codename = "noble";
      architecture = "armhf";
      url = "http://ports.ubuntu.com/ubuntu-ports/";
      operatingSystem = "ubuntu";
      packages_ = "libfuse3-3,libc6-dbg";
      outputHash = "sha256-9lnZXji215GkDSld20frHTKCcs4JjTnR5Qo+ojYhY34=";
    };
    "rootfs/ubuntu-24-04-aarch64" = mkRootfs {
      name = "ubuntu-24-04-aarch64";
      codename = "noble";
      architecture = "arm64";
      url = "http://ports.ubuntu.com/ubuntu-ports/";
      operatingSystem = "ubuntu";
      packages_ = "libfuse3-3,libc6-dbg";
      outputHash = "sha256-aqXwLdZaIy8k1vng+CJZDuqIWAT/ZgYTDhfdVV02qaU=";
    };
    "rootfs/ubuntu-24-04-s390x" = mkRootfs {
      name = "ubuntu-24-04-s390x";
      codename = "noble";
      architecture = "s390x";
      url = "http://ports.ubuntu.com/ubuntu-ports/";
      operatingSystem = "ubuntu";
      packages_ = "libfuse3-3,libc6-dbg";
      outputHash = "sha256-B1EUfnGgjjvrgczKWkEB6x7hVuMfF0fxvdaNOyTw54k=";
    };
    "rootfs/debian-bookworm-mipsel" = mkRootfs {
      name = "debian-bookworm-mipsel";
      codename = "bookworm";
      architecture = "mipsel";
      url = "https://ftp.debian.org/debian/";
      operatingSystem = "debian";
      packages_ = "libfuse2,libc6-dbg";
      outputHash = "sha256-fRNsTOy8y/d2ZySfYbh6BDhxgCrx3rQ7Ed/WYBptg3o=";
    };
    "rootfs/debian-buster-mips" = mkRootfs {
      name = "debian-buster-mips";
      codename = "buster";
      architecture = "mips";
      url = "https://archive.debian.org/debian/";
      operatingSystem = "debian";
      packages_ = "libfuse2,libc6-dbg";
      outputHash = "sha256-YWE1mOQ/JEbkEXDybD7WI9TGaz+158BpcN4Qd7lxFe8=";
    };
  };

  debugInfos = {
    "rootfs/ubuntu-20-04-x86-64/debug-info" = mkRootfsDebugInfo {
      name = "ubuntu-20-04-x86-64";
      rootfs = rootfses."rootfs/ubuntu-20-04-x86-64";
      outputHash = "sha256-IoGG9TRLdC74lM6tAFAItimuUa2xCrfbvfzBT+1OPi0=";
    };
    "rootfs/ubuntu-22-04-x86-64/debug-info" = mkRootfsDebugInfo {
      name = "ubuntu-22-04-x86-64";
      rootfs = rootfses."rootfs/ubuntu-22-04-x86-64";
      outputHash = "sha256-+xhgcoOyRyz7fKiKu/XeSjc0jBsVSFdX7Y0bOEzPgF0=";
    };
    "rootfs/ubuntu-24-04-x86-64/debug-info" = mkRootfsDebugInfo {
      name = "ubuntu-24-04-x86-64";
      rootfs = rootfses."rootfs/ubuntu-24-04-x86-64";
      outputHash = "sha256-IzW2wUoYjhYtrQl/JsAfKE5e3rDQaW+WzzVW6nY7wf8=";
    };
    "rootfs/ubuntu-24-04-i386/debug-info" = mkRootfsDebugInfo {
      name = "ubuntu-24-04-i386";
      rootfs = rootfses."rootfs/ubuntu-24-04-i386";
      outputHash = "sha256-JM+G59Co3lK3Hdhba2l3GlMEDjpB9+5ol+CKeMhy9qs=";
    };
    "rootfs/ubuntu-24-04-arm/debug-info" = mkRootfsDebugInfo {
      name = "ubuntu-24-04-arm";
      rootfs = rootfses."rootfs/ubuntu-24-04-arm";
      outputHash = "sha256-yE/h9EcAd80zsWn6j8ASn1I1YNHHB/meJfjssi+vMGE=";
    };
    "rootfs/ubuntu-24-04-aarch64/debug-info" = mkRootfsDebugInfo {
      name = "ubuntu-24-04-aarch64";
      rootfs = rootfses."rootfs/ubuntu-24-04-aarch64";
      outputHash = "sha256-zHhhil+y/R4x1mRtSez7PiUBwowBbDxljtNe7yCjRjc=";
    };
    "rootfs/ubuntu-24-04-s390x/debug-info" = mkRootfsDebugInfo {
      name = "ubuntu-24-04-s390x";
      rootfs = rootfses."rootfs/ubuntu-24-04-s390x";
      outputHash = "sha256-am9w70s8nzySduLsN/djEbV0QUeqk61Q2UzYqniUhK0=";
    };
    "rootfs/debian-bookworm-mipsel/debug-info" = mkRootfsDebugInfo {
      name = "debian-bookworm-mipsel";
      rootfs = rootfses."rootfs/debian-bookworm-mipsel";
      outputHash = "sha256-XiPB8RDdzI+SN+g/V5Imcul625ipfcZlZoBiikBk3tw=";
    };
    "rootfs/debian-buster-mips/debug-info" = mkRootfsDebugInfo {
      name = "debian-buster-mips";
      rootfs = rootfses."rootfs/debian-buster-mips";
      outputHash = "sha256-ftYboAIXtLV7KduuPtf3TvI99PbTbaYvk2GEmt6s8NI=";
    };
  };

  models = {
    "rootfs/ubuntu-20-04-x86-64/models" = mkRootfsModels {
      name = "ubuntu-20-04-x86-64";
      architecture = "x86_64";
      rootfs = rootfses."rootfs/ubuntu-20-04-x86-64";
      debugInfo = debugInfos."rootfs/ubuntu-20-04-x86-64/debug-info";
    };
    "rootfs/ubuntu-22-04-x86-64/models" = mkRootfsModels {
      name = "ubuntu-22-04-x86-64";
      architecture = "x86_64";
      rootfs = rootfses."rootfs/ubuntu-22-04-x86-64";
      debugInfo = debugInfos."rootfs/ubuntu-22-04-x86-64/debug-info";
    };
    "rootfs/ubuntu-24-04-x86-64/models" = mkRootfsModels {
      name = "ubuntu-24-04-x86-64";
      architecture = "x86_64";
      rootfs = rootfses."rootfs/ubuntu-24-04-x86-64";
      debugInfo = debugInfos."rootfs/ubuntu-24-04-x86-64/debug-info";
    };
    "rootfs/ubuntu-24-04-i386/models" = mkRootfsModels {
      name = "ubuntu-24-04-i386";
      architecture = "x86";
      rootfs = rootfses."rootfs/ubuntu-24-04-i386";
      debugInfo = debugInfos."rootfs/ubuntu-24-04-i386/debug-info";
    };
    "rootfs/ubuntu-24-04-arm/models" = mkRootfsModels {
      name = "ubuntu-24-04-arm";
      architecture = "arm";
      rootfs = rootfses."rootfs/ubuntu-24-04-arm";
      debugInfo = debugInfos."rootfs/ubuntu-24-04-arm/debug-info";
    };
    "rootfs/ubuntu-24-04-aarch64/models" = mkRootfsModels {
      name = "ubuntu-24-04-aarch64";
      architecture = "aarch64";
      rootfs = rootfses."rootfs/ubuntu-24-04-aarch64";
      debugInfo = debugInfos."rootfs/ubuntu-24-04-aarch64/debug-info";
    };
    "rootfs/ubuntu-24-04-s390x/models" = mkRootfsModels {
      name = "ubuntu-24-04-s390x";
      architecture = "systemz";
      rootfs = rootfses."rootfs/ubuntu-24-04-s390x";
      debugInfo = debugInfos."rootfs/ubuntu-24-04-s390x/debug-info";
    };
    "rootfs/debian-bookworm-mipsel/models" = mkRootfsModels {
      name = "debian-bookworm-mipsel";
      architecture = "mipsel";
      rootfs = rootfses."rootfs/debian-bookworm-mipsel";
      debugInfo = debugInfos."rootfs/debian-bookworm-mipsel/debug-info";
    };
    "rootfs/debian-buster-mips/models" = mkRootfsModels {
      name = "debian-buster-mips";
      architecture = "mips";
      rootfs = rootfses."rootfs/debian-buster-mips";
      debugInfo = debugInfos."rootfs/debian-buster-mips/debug-info";
    };
  };
in
rootfses // debugInfos // models
