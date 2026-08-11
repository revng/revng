{ pkgs, stdenv, revngPackages }:
let
  src = pkgs.fetchurl {
    url = "https://github.com/tpoechtrager/cctools-port/archive/cctools-877.8-ld64-253.9-1.tar.gz";
    hash = "sha256-yIsGMbHXu1GG3WRmpi9SINxhkfKy2cfBIrMnOF5zSq8=";
  };

  libtapi = revngPackages."macos/libtapi";

  mkLd64 = { triple }:
    stdenv.mkDerivation {
      name = "macos-ld64-${triple}";
      inherit src;

      nativeBuildInputs = with pkgs; [
        autoconf
        automake
        libtool
        pkg-config
        clang
      ];

      buildInputs = with pkgs; [
        openssl
        libuuid
      ];

      patchPhase = ''
        # The Blob<>::clone() template calls a non-existent
        # BlobCore::clone() — older clang accepted this since the
        # template was never instantiated; newer clang catches it
        # at template definition time. Drop the dead method.
        sed -i '/BlobType \*clone() const$/,/{ assert(validateBlob()); return specific(this->BlobCore::clone());\t}$/d' \
          cctools/ld64/src/ld/code-sign-blobs/blob.h
      '';

      configurePhase = ''
        cd cctools

        # sys/sysctl.h was dropped from glibc 2.30; cctools-port only
        # uses it to autodetect the macOS deployment target. Replace
        # the foreign header with a stub that makes sysctl() return -1,
        # which already triggers the "fall back to default" branch.
        cat > include/foreign/sys/sysctl.h <<'STUB'
        #ifndef _CCTOOLS_SYSCTL_STUB
        #define _CCTOOLS_SYSCTL_STUB
        #include <sys/types.h>
        #include <errno.h>
        #define CTL_KERN 1
        #define KERN_OSRELEASE 2
        static inline int sysctl(int *name, unsigned int namelen,
                                 void *oldp, size_t *oldlenp,
                                 void *newp, size_t newlen) {
          (void)name; (void)namelen; (void)oldp; (void)oldlenp;
          (void)newp; (void)newlen;
          errno = ENOSYS;
          return -1;
        }
        #endif
        STUB

        autoreconf -fi

        # newer clang surfaces -Wsometimes-uninitialized and friends
        # that the 2016 cctools tree expected to be silent.
        find . -name Makefile.in -exec sed -i 's|-Werror||g' {} \;

        ./configure \
          --prefix=$out \
          --target=${triple} \
          --enable-tapi-support \
          CC=clang \
          CXX=clang++ \
          OBJC=clang \
          CXXFLAGS="-I${libtapi}/include -Wno-error" \
          LDFLAGS="-L${libtapi}/lib -fuse-ld=bfd -Wl,--allow-multiple-definition"
      '';

      enableParallelBuilding = true;
    };
in
{
  inherit mkLd64;
  "x86-64" = mkLd64 { triple = "x86_64-apple-darwin11"; };
  "i686" = mkLd64 { triple = "i686-apple-darwin11"; };
  arm = mkLd64 { triple = "arm-apple-darwin11"; };
  aarch64 = mkLd64 { triple = "aarch64-apple-darwin11"; };
}
