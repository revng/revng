{ pkgs, stdenv, revngPackages }:
let
  baseClang = pkgs.llvmPackages_21.clang-unwrapped;

  # Xcode 9's MacOSX.sdk predates Apple Silicon: its arch-dispatch
  # headers (sys/cdefs.h, machine/_types.h, machine/endian.h, …) only
  # know __i386__ and __x86_64__, so any aarch64-apple-darwin11 build
  # hits "#error Unsupported architecture" once it pulls in the SDK.
  # revng-qa's apple ABI tests only need the freestanding parts of
  # libc — clang already supplies <stdint.h>/<stddef.h>/<stdarg.h>; we
  # ship the missing <assert.h>/<string.h> as forward declarations and
  # skip --sysroot. The mach-o objects then build without touching the
  # SDK's incompatible system headers.
  freestandingHeaders = pkgs.runCommand "macos-freestanding-headers" { } ''
    mkdir -p $out/include
    cat > $out/include/assert.h <<'EOF'
    #ifndef _CCTOOLS_ASSERT_STUB
    #define _CCTOOLS_ASSERT_STUB
    #define assert(x) ((void)0)
    #endif
    EOF
    cat > $out/include/string.h <<'EOF'
    #ifndef _CCTOOLS_STRING_STUB
    #define _CCTOOLS_STRING_STUB
    #include <stddef.h>
    void *memcpy(void *, const void *, size_t);
    void *memmove(void *, const void *, size_t);
    void *memset(void *, int, size_t);
    int   memcmp(const void *, const void *, size_t);
    size_t strlen(const char *);
    #endif
    EOF
  '';

  mkClang = { triple, ld64 }:
    pkgs.runCommand "macos-clang-${triple}" {
      passthru = { inherit triple; };
    } ''
      mkdir -p $out/bin
      for tool in clang clang++; do
        cat > $out/bin/${triple}-$tool <<EOF
      #!${pkgs.runtimeShell}
      exec -a ${triple}-$tool \
        ${baseClang}/bin/$tool \
        -isystem ${freestandingHeaders}/include \
        -B${ld64}/bin \
        "\$@"
      EOF
        chmod +x $out/bin/${triple}-$tool
      done
    '';
in
{
  inherit mkClang;
  "x86-64" = mkClang {
    triple = "x86_64-apple-darwin11";
    ld64 = revngPackages."macos/ld64/x86-64";
  };
  "i686" = mkClang {
    triple = "i686-apple-darwin11";
    ld64 = revngPackages."macos/ld64/i686";
  };
  arm = mkClang {
    triple = "arm-apple-darwin11";
    ld64 = revngPackages."macos/ld64/arm";
  };
  aarch64 = mkClang {
    triple = "aarch64-apple-darwin11";
    ld64 = revngPackages."macos/ld64/aarch64";
  };
}
