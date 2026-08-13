{ pkgs, pkgs-2505, python }:
let
  makeQemu =
    pkgs: llvmPackages: name: cflags: suffixes:
    (llvmPackages.stdenv.mkDerivation {
      name = name;

      src = pkgs.fetchFromGitHub {
        owner = "revng";
        repo = "qemu";
        # Match orchestra (extracted from
        # `.orchestra/binary-archives/origin/linux-x86-64/qemu-helpers/
        # optimized/8035324…_*.hash-material.yml`). This is a downstream
        # rev.ng patch series that, among other things, refactors the
        # `comis_eflags[ret + 1]` array access inside `helper_(u)comi(s|s)d`
        # into a callable `lookup_comis_eflags(int)` helper — keeping the
        # GEP out of `helper_comisd`'s body so revng's
        # `detect-uninlinable-helpers` doesn't demote the `revng_inline`
        # tag and the lifted IR keeps `call @float64_compare` /
        # `call @float32_compare` visible to the floating-point-x86-64
        # FileCheck.
        rev = "8035324196ca7f2d63c63deae1b0e38987573789";
        hash = "sha256-mq4KJezaHWCOVV5qmxgbG+xLTEDzfP7tbchFtV5ocss=";
      };

      postPatch = ''
        patchShebangs python/scripts/link-embedded-objects

        grep -vF "subdir('fp')" tests/meson.build > tests/meson.build2
        mv tests/meson.build2 tests/meson.build

        # WIP
        grep -vF "_Static_assert" target/i386/cpu.h > target/i386/cpu.h2
        mv target/i386/cpu.h2 target/i386/cpu.h

        grep -vF "ASSERT_CONSTANT" libtcg/libtcg.c > libtcg/libtcg.c2
        mv libtcg/libtcg.c2 libtcg/libtcg.c
      '';

      preBuild = ''
        cd build
      '';

      nativeBuildInputs = (with pkgs; [
        pkg-config
        meson
        ninja
        coreutils-full
        llvmPackages.clang
        llvmPackages.llvm
      ]) ++ [
        (python.withPackages (python-pkgs: [ python-pkgs.distlib ]))
        # Hooks from the python package are needed to add `$pythonPath` so
        # `python/scripts/mkvenv.py` can detect `meson` otherwise the vendored meson without patches will be used.
        python.pkgs.python
      ];

      buildInputs = with pkgs; [
        glib
        zlib
      ];

      dontUseMesonConfigure = true;
      enableParallelBuilding = true;

      configureFlags =
        let
          targets = builtins.concatStringsSep "," (
            pkgs.lib.flatten (
              map (
                suffix:
                map (architecture: "${architecture}-${suffix}") [
                  "arm"
                  "aarch64"
                  "i386"
                  "mips"
                  "mipsel"
                  "s390x"
                  "x86_64"
                ]
              ) suffixes
            )
          );
        in
        [
          "--disable-plugins"
          "--target-list=${targets}"
          "--disable-werror"
          "--disable-docs"
          "--disable-kvm"
          "--disable-tools"
          "--disable-system"
          "--disable-libnfs"
          "--disable-vde"
          "--disable-gnutls"
          "--disable-cap-ng"
          "--disable-pie"
          "-Dvhost_user=disabled"
          "-Dxkbcommon=disabled"
          "--extra-cflags=-Wno-unused-variable"
          "--extra-cflags=-Wno-unused-function"
          "--extra-cflags=-Wno-unused-result"
          "--extra-cflags=-Wno-unused-but-set-variable"
          (map (argument: "--extra-cflags=${argument}") cflags)
        ];

      preInstall = ''
        mkdir -p $out/include
        mkdir -p $out/lib
      '';

      # The qemu develop branch dropped the glib entry from
      # libtcg-*.so's RUNPATH; revng dlopens these at build time and
      # the loader then can't find libglib-2.0.so.0. Add it back.
      # Also stamp zlib+glib into every qemu-* binary's RPATH so the
      # linux-user runtimes don't resolve libz / libglib against the
      # host filesystem (which was leaking on Ubuntu 18.04 and picking
      # up a libc.so.6 that doesn't have GLIBC_2.35).
      postFixup = ''
        for f in $out/lib/libtcg-*.so $out/bin/qemu-*; do
          [ -f "$f" ] || continue
          current=$(patchelf --print-rpath "$f" 2>/dev/null || true)
          patchelf --set-rpath "${pkgs.glib.out}/lib:${pkgs.zlib.out}/lib''${current:+:$current}" "$f"
        done
      '';
    });
in
{
  # Build our fork of QEMU
  qemu = makeQemu pkgs pkgs.llvmPackages_21 "qemu" [ "-fPIC" ] [ "linux-user" "libtcg" ];
  qemuHelpers =
    makeQemu pkgs-2505 pkgs-2505.llvmPackages_16 "qemu-helpers"
      [
        "-fPIC"
        "-Wno-gcc-compat"
        "-DGEN_LLVM_HELPERS"
        "-O0"
        "-Xclang"
        "-disable-O0-optnone"
        "-fembed-bitcode"
      ]
      [ "llvm-helpers" ];
}
