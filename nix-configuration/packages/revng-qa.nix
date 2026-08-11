{ pkgs, stdenv, python, crossToolchains, msvc, ninjaShellRule, inputs, revngPackages }:
let
  revng-qa = stdenv.mkDerivation {
    name = "revng-qa";

    src = inputs.revng-qa;

    nativeBuildInputs = (with pkgs; [
      cmake
      ninja
    ]) ++ [
      (python.withPackages (
        ps: with ps; [
          jinja2
          pyyaml
        ]
      ))
    ];

    cmakeFlags = [
      "-GNinja"
    ];

    # `aarch64-unknown-linux-musl-ld` (binutils 2.35) emits a
    # PT_LOAD segment for BSS whose `p_offset` is chosen to satisfy
    # the 64K page alignment (0xfe8), then falls past the end of
    # the tiny `nostdlib` binary. The old `llvm-objcopy` tolerated
    # it; the bumped LLVM refuses with "program header ... goes
    # past the end of the file". Adding `-Wl,-N` (--omagic) drops
    # the page-alignment of data so the BSS PT_LOAD's file offset
    # stays inside the file. This mirrors what the `arm` arch
    # already does (`-Wl,-Ttext-segment=…`).
    postPatch = ''
      substituteInPlace share/revng/test/configuration/revng-qa/architectures.yml \
        --replace-fail $'      QEMU_NAME: aarch64\n      COMMON_CFLAGS:' \
        $'      QEMU_NAME: aarch64\n      GCC_CFLAGS:\n        - -Wl,-N\n      COMMON_CFLAGS:'
    '';

  };

  testRevngQa = stdenv.mkDerivation {
    name = "test/revng-qa";

    __structuredAttrs = true;
    unsafeDiscardReferences.out = true;

    unpackPhase = "true";

    # The host stdenv's setup-hook sets `NIX_HARDENING_ENABLE` to its
    # full default, and every cross-compiler wrapper's setup-hook
    # uses `: ${VAR=default}` semantics — i.e. it keeps the host's
    # value rather than its own baked-in `defaultHardeningFlags`. So
    # the only effective way to trim the cross-compile hardening is
    # via `hardeningDisable` on this derivation.
    #
    # Drop:
    #   - pic: `-fPIC` injects GOT round-trips for static globals on
    #     x86-64 (`mov 0x0(%rip),%rax; movss (%rax),%xmm0`) where
    #     orchestra's plain gcc-9.2.0 emits a direct RIP-relative
    #     load. That extra dereference reshapes revng's CSV access
    #     pattern enough for DetectABI to drop XMM-register inputs
    #     and skip the helper inlining that produces
    #     `@float32_compare`.
    #   - stackclashprotection / stackprotector: extra prologue/
    #     epilogue code; MIPS softfloat-conversion lifter
    #     (cvt.w.d/mfc1) abandons the function before the conversion.
    #   - fortify / fortify3: runtime checks around libc calls; we
    #     don't link libc but the macro still alters inline expansions.
    #   - zerocallusedregs: `xor %eax,%eax; xor %edx,%edx` before
    #     every `ret`; DetectABI misreads the zeroed GPRs as live-out.
    # Disable every hardening flag for the cross-toolchains *except*
    # `pie`: the cross-compiler wrappers in `nativeBuildInputs` consult
    # the bare `NIX_HARDENING_ENABLE` (host role), so trimming it strips
    # every wrapper-injected flag (`-fPIC`, `-Wformat-security`,
    # `-fno-strict-overflow`, …), letting binaries match what
    # orchestra's gentoo-style gcc emits — which is what revng's
    # per-arch lifters were written against.
    #
    # Historical note on `pic`: when the `revng-qa.compiled-stripped`
    # rule used `llvm-objcopy --strip-all`, dropping `pic` on aarch64
    # produced a binary with a BSS-only LOAD segment (`FileSize=0`,
    # `offset=0xfe8`) that `llvm-objcopy` rejected — the ELF was
    # spec-compliant (BSS doesn't need file bytes), but `llvm-objcopy`
    # is over-strict. The rule now calls `${TRIPLE}objcopy` (GNU
    # binutils) which accepts the segment, so `pic` can be dropped.
    hardeningDisable = [
      "bindnow"
      "format"
      "fortify"
      "fortify3"
      "nostrictaliasing"
      "pacret"
      "pic"
      "relro"
      "shadowstack"
      "stackclashprotection"
      "stackprotector"
      "strictoverflow"
      "trivialautovarinit"
      "zerocallusedregs"
    ];

    # Our custom nixpkgs branch (`feature/improve-uclibc-ng`) bakes
    # `-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer` into
    # every gcc-wrapper's `cc-cflags-before` for non-x86_32 / non-s390
    # targets (see `pkgs/build-support/cc-wrapper/default.nix`).
    # orchestra's plain gcc-9.2.0 emits a frame pointer by default at
    # `-O0`, which is what revng's lifters were trained on — but we
    # still want to flip it for the `revng-qa.compiled` targets here,
    # for one orthogonal reason: our cross-toolchain's `libcCross`
    # (musl) is built by the host gcc 14.3.0 rather than the cross
    # gcc 9.2.0 (see `nix/packages/cross-toolchains.nix` — `wrapCCWith`
    # wraps only the cross gcc binary, not `libcCross`). gcc-14's
    # IPA-SRA splits musl's `pad()` into a `pad.part.0` clone with a
    # narrower signature, and with frame pointer on, the resulting
    # epilogue (`mov -0x8(%rbp), %rbx; leave; ret`) buries the last
    # `%rax` write three instructions deep — DetectABI then drops the
    # return value and the SegregateStackAccesses test fails. Omitting
    # the frame pointer here gives a clean `pop %rbx; ret` epilogue
    # that DetectABI handles. Override via `NIX_CFLAGS_COMPILE`, which
    # the wrapper appends *after* `cc-cflags-before`, so this wins.
    NIX_CFLAGS_COMPILE = "-fomit-frame-pointer";


    nativeBuildInputs =
      with pkgs;
      (
        [
          binutils
          gcc
          llvm_21
          lld_21
        ]
        ++ crossToolchains
      )
      ++ msvc.toolchains
      ++ [
        revngPackages."macos/clang/x86-64"
        revngPackages."macos/clang/i686"
        revngPackages."macos/clang/arm"
        revngPackages."macos/clang/aarch64"
      ]
      ++ [
        revng-qa
        ninjaShellRule
        (python.withPackages (
          ps: with ps; [
            jinja2
            pyyaml
          ]
        ))
        # WIP: this should be pulled by MSVC dep
        pkgs.samba
      ];

    buildPhase = ''
      echo
    '';

    installPhase = ''
      mkdir -p $out
      python3 \
        ${revng-qa}/libexec/revng/test-configure \
        "${revng-qa}/share/revng/test/configuration/revng-qa/"*.yml \
        --install-path "${revng-qa}" \
        --destination . \
        --target-type 'revng-qa\..*'
      export REVNG_OPTIONS="--debug-log=verify"
      # test-configure emits `shell = /bin/bash` on every rule; that
      # path doesn't exist inside the nix sandbox. Point ninja at
      # the bash we actually have.
      sed -i "s|shell = /bin/bash|shell = ${pkgs.bash}/bin/bash|g" build.ninja
      export XDG_CACHE_HOME="$PWD/.cache"
      mkdir -p "$XDG_CACHE_HOME/.cache"
      mkdir -p extra-includes/gnu

      i386-winsdk-vc12-cl || true
      i386-winsdk-vc13-cl || true
      i386-winsdk-vc16-cl || true
      i386-winsdk-vc19-cl || true
      x86_64-winsdk-vc19-cl || true
      aarch64-winsdk-vc19-cl || true

      cp -a ${pkgs.glibc.dev}/include/gnu/stubs-64.h extra-includes/gnu/stubs-32.h
      # stdenv's mold-linker setup exports NIX_CFLAGS_LINK / NIX_LDFLAGS
      # that the cross-toolchains pick up and then fail to resolve at
      # link time. Clear them so each cross-gcc finds its own ld.
      NIX_CFLAGS_COMPILE="$NIX_CFLAGS_COMPILE -isystem$PWD/extra-includes" \
        NIX_CFLAGS_LINK= \
        ninja all
      # Copy the built test artifacts into $out so downstream
      # derivations (test/revng) can consume them. The build
      # graph put them under share/ relative to the build dir.
      mkdir -p "$out/share"
      cp -a share/revng "$out/share/"
    '';

  };
in
{
  inherit revng-qa;
  "test/revng-qa" = testRevngQa;
}
