{ nixpkgsOld, nixpkgsArm, system }:
let
  # `pkgs` (used for `runCommand`, `fetchurl`, lib helpers) doesn't
  # need to be the same nixpkgs every cross-compiler is built from —
  # use the old one for consistency with the bulk of the toolchains.
  pkgs = import nixpkgsOld { inherit system; };

  # Build a thin derivation whose bin/ exposes every `${from}-X` binary
  # from `cc` as `${to}-X`. Used because nixpkgs only accepts a small
  # set of vendors (pc/unknown/...), but revng-qa expects gentoo/ibm/
  # hardfloat-flavored triples.
  tripleAlias =
    { from, to, cc }:
    pkgs.runCommand "${to}-cc-alias" { } ''
      mkdir -p $out/bin
      for f in ${cc}/bin/${from}-*; do
        ln -s "$f" "$out/bin/${to}-''${f##*/${from}-}"
      done
    '';

  # Overlay used by `crossCompilerFrom` to pin `musl` to the version
  # orchestra ships (1.1.12 for x86_64/mips*/i386, 1.1.19 for
  # aarch64/s390x). Nixpkgs ships 1.2.5, but revng's DetectABI is
  # trained against the earlier musl's startup + libc code paths —
  # notably `__libc_start_main` (K&R 3-arg definition vs. 6-arg
  # declaration in 1.1.x, unified to 6-arg in 1.2.x) — and a bump
  # changes the emerging function prototypes enough to make
  # `test-segregate-stack-accesses` and `test-dla` diverge from
  # their pinned expected models. Drop the nixpkgs-specific CVE /
  # stdio patches (they don't apply against 1.1.x source), keep the
  # openwrt relative-symlink patch.
  overrideMusl =
    muslVersion: muslHash:
    (final: prev: {
      musl = prev.musl.overrideAttrs (old: {
        version = muslVersion;
        src = pkgs.fetchurl {
          url = "https://musl.libc.org/releases/musl-${muslVersion}.tar.gz";
          hash = muslHash;
        };
        patches = builtins.filter (
          p: pkgs.lib.hasSuffix "300-relative.patch" (toString p)
        ) (old.patches or [ ]);
        # musl 1.1.x's configure doesn't grow its own AR/RANLIB
        # accessor knobs; it derives them from CROSS_COMPILE. The
        # nixpkgs stdenv exposes only prefixed `${triple}-ar`, so
        # without CROSS_COMPILE the Makefile falls back to the plain
        # `ar` name which isn't on PATH. Set it from the target triple.
        configureFlags = (old.configureFlags or [ ]) ++ [
          "CROSS_COMPILE=${final.stdenv.hostPlatform.config}-"
        ];
      });
    });

  crossCompilerFrom = (
    nixpkgs: triple: binutilsVersion: binutilsHash: gccVersion: muslVersion: muslHash:
    let
      crossNixpkgs = import nixpkgs {
        crossSystem = {
          config = triple;
        };
        inherit system;
        overlays = pkgs.lib.optional (
          pkgs.lib.hasSuffix "musl" triple && muslVersion != null
        ) (overrideMusl muslVersion muslHash);
      };
      pkgs2 = crossNixpkgs.buildPackages;
      binutils_old_real = pkgs2.bintools.bintools.overrideAttrs {
        version = binutilsVersion;
        patches = [ ];
        outputs = [
          "out"
          "info"
          "man"
          "dev"
        ]; # Omit lib

        src = pkgs.fetchurl {
          url = "mirror://gnu/binutils/binutils-${binutilsVersion}.tar.bz2";
          hash = binutilsHash;
        };
      };
      binutils_old = pkgs2.wrapBintoolsWith {
        bintools = binutils_old_real;
        # Bake a trimmed hardening default into the wrapper so each
        # cross-gcc invocation sees the right NIX_HARDENING_ENABLE_*
        # at *compile time*. Doing this in test/revng-qa's preBuild
        # is too late: the wrapper's setup-hook mangles
        # `NIX_HARDENING_ENABLE` → `NIX_HARDENING_ENABLE_<salt>` at
        # shell-init, well before preBuild runs.
        #
        # Drop:
        #   - fortify / fortify3 — runtime-check inlining.
        #   - pic — `-fPIC` makes gcc emit GOT-relative loads
        #     (`mov rip+disp,%rax; movss (%rax),%xmm0`) where
        #     orchestra emits direct RIP-relative (`movss
        #     0x...(%rip),%xmm0`). That extra dereference reshapes
        #     the function's CSV access pattern enough for revng's
        #     DetectABI to drop the XMM-register inputs and skip
        #     helper inlining for `@float32_compare` etc.
        #   - stackclashprotection / stackprotector — extra
        #     prologue/epilogue code; MIPS softfloat-conversion
        #     lifter (cvt.w.d/mfc1) ends up abandoning the function
        #     before reaching the conversion.
        #   - zerocallusedregs — `xor %eax,%eax; xor %edx,%edx`
        #     before every `ret`; DetectABI misreads the zeroed
        #     GPRs as part of the live-out set.
        #
        # Keep `pie` even though we don't actively need it; dropping
        # it on gcc-9.2 aarch64 emits a malformed program header
        # that `llvm-objcopy --strip-all` rejects.
        defaultHardeningFlags = [
          "bindnow"
          "format"
          "pie"
          "relro"
          "strictoverflow"
        ];
      };
      pkgs3 = pkgs2 // {
        #binutils = binutils_old;
        #bintools = binutils_old;

        # NOTE: we might want to use binutils_old for stdenvNoLibc as well, whih
        # is currently otherwise built with non-custom versions of GCC/binutils
        stdenv = pkgs2.stdenv // {
          cc = pkgs2.stdenv.cc // {
            bintools = binutils_old;
          };
        };

        # TODO: we're not applying uclibc configuration. The following does not work:
        # stdenvNoLibc = pkgs2.stdenvNoLibc // {
        #   hostPlatform = pkgs2.hostPlatform // {
        #     uclibc = pkgs2.hostPlatform.uclibc // {
        #       extraConfig = "${builtins.readFile ./uClibc.config}";
        #     };
        #   };
        # };
      };
      # nixpkgs-2505's gcc-9 ships 9.5.0, but revng's MIPS lifter
      # was written against the codegen patterns gcc-9.2.0 emits
      # (cvt.w.d followed by mfc1+sw rather than swc1, plus a stack
      # frame even at -O0). Override the version + src so the cross-
      # compilers produce binaries the lifter recognises. Only kicks
      # in when gccVersion is "9.2.0" — other versions fall through.
      ccBase = pkgs3.callPackage (nixpkgs + "/pkgs/development/compilers/gcc/default.nix") (
        {
          # When asking for 9.2.0 we still pin majorMinorVersion to
          # "9" so the upstream wiring picks the gcc-9 codepaths; the
          # exact 9.2.0 vs 9.5.0 swap happens via `overrideAttrs`
          # below.
          majorMinorVersion =
            if pkgs.lib.hasPrefix "9." gccVersion then "9" else gccVersion;
          noSysDirs = true;
          #binutils = binutils_old;
          targetPackages = pkgs3;
          # nixpkgs-2505 exposes the right cross libc directly as
          # `pkgs.libcCross` (musl / glibc / uClibc picked by target
          # platform). On the custom nixpkgs branch with the gcc-ng
          # rework, `targetPackages.libc` is the equivalent.
          libcCross = pkgs3.libcCross or pkgs3.targetPackages.libc;
        }
        // (
          if pkgs.lib.hasInfix "mingw" triple then
            {
              threadsCross = {
                model = "win32";
                package = null;
              };
            }
          else
            { }
        )
      );
      cc =
        if gccVersion == "9.2.0" then
          ccBase.overrideAttrs (old: {
            version = "9.2.0";
            src = pkgs.fetchurl {
              url = "mirror://gcc/releases/gcc-9.2.0/gcc-9.2.0.tar.xz";
              sha256 = "sha256-6m7wjxISOdpWlfdsmzNjehGNz2PiQWRCIjGRf6YfsgY=";
            };
            # nixpkgs-2505 unconditionally applies a 9.5.0-era
            # libgcc/config/aarch64/lse.S patch
            # (`cfi_startproc-reorder-label-09-1.diff`) for any gcc <
            # 14. That file didn't exist in 9.2.0, so the patch fails
            # to apply. Drop it from the list.
            patches = builtins.filter (
              p:
              !pkgs.lib.hasSuffix "cfi_startproc-reorder-label-09-1.diff" (toString p)
            ) (old.patches or [ ]);
            # nixpkgs's gcc postPatch has a `baseVersion ==
            # gcc/BASE-VER` sanity check using the nix-eval-bound
            # baseVersion (= 9.5.0 from majorMinorVersion="9"); our
            # 9.2.0 source trips it. By the time `postPatch` is a
            # string, `${baseVersion}` is already interpolated to
            # `9.5.0`, so substitute the literal.
            postPatch = pkgs.lib.replaceStrings
              [ "[[ 9.5.0 != $gcc_base_version ]]" ]
              [ "[[ 9.2.0 != $gcc_base_version ]]" ]
              (old.postPatch or "");
          })
        else
          ccBase;
    in
    (pkgs3.wrapCCWith {
      inherit cc;
      bintools = binutils_old;
    })
  );

  # All cross-compilers except armv7a pull from `nixpkgsOld` for the
  # gcc 9 toolchain. armv7a uses the primary nixpkgs.
  crossCompiler = crossCompilerFrom nixpkgsOld;
  crossCompilerArm = crossCompilerFrom nixpkgsArm;
in
[
  # crossNixpkgs.buildPackages.gcc9
  # (crossNixpkgs.buildPackages.gcc9.override {
  #   bintools = crossNixpkgs.buildPackages.gcc9.bintools.overrideAttrs {
  #     version = "2.35";
  #     bintools = crossNixpkgs.buildPackages.gcc9.bintools.bintools.overrideAttrs {
  #       version = "2.35";
  #       patches = [];
  #       outputs = [
  #         "out"
  #         "info"
  #         "man"
  #         "dev"
  #       ]; # Omit lib

  #       src = pkgs.fetchurl {
  #         url = "mirror://gnu/binutils/binutils-2.35.tar.bz2";
  #         hash = "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c=";
  #       };
  #     };
  #   };
  # })

  # (
  #   crossNixpkgs.buildPackages.pkgs.wrapCC (
  #     crossNixpkgs.buildPackages.pkgs.callPackage (<nixpkgs> + "/pkgs/development/compilers/gcc/default.nix") {
  #       majorMinorVersion = "9";
  #       noSysDirs = true;
  #     }
  #   )
  # )

  # TODO: pin musl version, we wanted 1.1.12, we're getting 1.2.5
  # TODO: pin uclibc version, we wanted ???, we're getting ???

  # Orchestra pins gcc-9.2.0 for mips/mipsel/i386/x86-64/arm and
  # gcc-7.3.0 for aarch64/s390x. revng's MIPS lifter (and other
  # backends) depend on specific instruction patterns the compiler
  # emits; bumping to gcc 13 changes the emitted code enough to make
  # the lifter abandon some functions before reaching the conversion
  # instructions, breaking FileCheck-based tests (floating-point,
  # CollectCFG, segregate-stack-accesses, pypeline-comparison, ...).
  # nixpkgs has 9.5.0 as `gcc9` and 7.5.0 as `gcc7`; orchestra is on
  # the 9.2.0/7.3.0 minor before, but the major matches.
  (crossCompiler "mips-unknown-linux-musl" "2.35"
    "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c="
    "9.2.0"
    "1.1.12"
    "sha256-cguDx+J2tLZ5wL/+lQk0DV+B/WAVCOYH5wgXffDTHA4="
  )
  (crossCompiler "mipsel-unknown-linux-musl" "2.35"
    "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c="
    "9.2.0"
    "1.1.12"
    "sha256-cguDx+J2tLZ5wL/+lQk0DV+B/WAVCOYH5wgXffDTHA4="
  )
  # Orchestra builds aarch64/s390x with gcc-7.3.0, but nixpkgs-2505
  # has already dropped gcc7 (lowest is gcc9). Fall back to gcc9 here
  # — same major as the rest of the toolchain. No F-family failures
  # land on these arches, so the minor compiler delta is acceptable.
  (crossCompiler "aarch64-unknown-linux-musl" "2.35"
    "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c="
    "9.2.0"
    "1.1.19"
    "sha256-21moV4ImuYNz9bJ+YfDdKa0kVvSqnOxYe6jCRQjkwdk="
  )
  # Build the cross-compilers with vendors nixpkgs knows (pc/unknown);
  # the vendor names revng-qa expects (gentoo/ibm/hardfloat) come in
  # as symlinks via `tripleAlias` below.
  #
  # armv7a-uclibceabihf: built from the primary nixpkgs (newer uClibc
  # 1.0.55 cross-builds; nixpkgs-2505's older 1.0.52 fails for this
  # triple with "toolchain was built for EABI, but you have chosen
  # OABI"). No F-family failure lands on arm so the compiler-major
  # delta vs orchestra (gcc 9.2.0) is acceptable here.
  (crossCompilerArm "armv7a-unknown-linux-uclibceabihf" "2.35"
    "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c="
    "13"
    null
    null
  )
  (crossCompiler "x86_64-unknown-linux-musl" "2.35"
    "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c="
    "9.2.0"
    "1.1.12"
    "sha256-cguDx+J2tLZ5wL/+lQk0DV+B/WAVCOYH5wgXffDTHA4="
  )
  (crossCompiler "i686-unknown-linux-musl" "2.35"
    "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c="
    "9.2.0"
    "1.1.12"
    "sha256-cguDx+J2tLZ5wL/+lQk0DV+B/WAVCOYH5wgXffDTHA4="
  )
  (crossCompiler "s390x-unknown-linux-musl" "2.35"
    "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c="
    "9.2.0"
    "1.1.19"
    "sha256-21moV4ImuYNz9bJ+YfDdKa0kVvSqnOxYe6jCRQjkwdk="
  )
  # mingw cross-compilers — no upstream-pinned version requirement.
  (crossCompiler "i686-w64-mingw32" "2.35" "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c=" "13" null null)
  (crossCompiler "x86_64-w64-mingw32" "2.35" "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c="
    "13" null null
  )

  # The bumped revng-qa expects gentoo/ibm/hardfloat-vendor triple
  # names. nixpkgs doesn't recognize those vendors, so expose them as
  # symlinks pointing at the binaries above.
  (tripleAlias {
    from = "armv7a-unknown-linux-uclibceabihf";
    to = "armv7a-hardfloat-linux-uclibceabi";
    cc = (crossCompilerArm "armv7a-unknown-linux-uclibceabihf" "2.35"
      "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c=" "13" null null);
  })
  (tripleAlias {
    from = "x86_64-unknown-linux-musl";
    to = "x86_64-gentoo-linux-musl";
    cc = (crossCompiler "x86_64-unknown-linux-musl" "2.35"
      "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c=" "9.2.0"
      "1.1.12" "sha256-cguDx+J2tLZ5wL/+lQk0DV+B/WAVCOYH5wgXffDTHA4=");
  })
  (tripleAlias {
    from = "i686-unknown-linux-musl";
    to = "i386-gentoo-linux-musl";
    cc = (crossCompiler "i686-unknown-linux-musl" "2.35"
      "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c=" "9.2.0"
      "1.1.12" "sha256-cguDx+J2tLZ5wL/+lQk0DV+B/WAVCOYH5wgXffDTHA4=");
  })
  (tripleAlias {
    from = "s390x-unknown-linux-musl";
    to = "s390x-ibm-linux-musl";
    cc = (crossCompiler "s390x-unknown-linux-musl" "2.35"
      "sha256-fSRmD4cJNnBzjli8x7ewbxIcD8sMqPxENo1nWl75z/c=" "9.2.0"
      "1.1.19" "sha256-21moV4ImuYNz9bJ+YfDdKa0kVvSqnOxYe6jCRQjkwdk=");
  })
]
