{ pkgs, pkgs-2505, stdenv }:
# Builds only `libtapi` from Apple's open-source clang-800.0.42.1
# tree, with tapi-1.30 dropped into projects/libtapi. We do NOT
# build a usable clang — the surrounding clang tree is just the
# CMake scaffolding tapi needs.
stdenv.mkDerivation {
  name = "macos-libtapi";

  srcs = [
    (pkgs.fetchurl {
      url = "https://opensource.apple.com/tarballs/clang/clang-800.0.42.1.tar.gz";
      hash = "sha256-cfvtZ5OzI8B6LZR6LqLYq7E5TnRw11R13d3lmxh+zrA=";
    })
    (pkgs.fetchurl {
      url = "https://opensource.apple.com/tarballs/tapi/tapi-1.30.tar.gz";
      hash = "sha256-sCWXIGU4/Jmjxor4OKPhfW3MCn8/h+bSHRTnGMQGrkI=";
    })
  ];

  nativeBuildInputs = [
    # The 2016-era clang tree still uses pre-3.5 cmake policies
    # (notably CMP0051 OLD) that cmake 4.x dropped support for.
    pkgs-2505.cmake
    pkgs.ninja
    pkgs.python3
  ];

  unpackPhase = ''
    # The clang tarball is clang-clang-800.0.42.1/src/CMakeLists.txt etc;
    # strip the outer dir so `src/` lands at top-level.
    tar --strip-components=1 -xzf $(echo $srcs | awk '{print $1}')
    mkdir -p src/projects/libtapi
    tar -C src/projects/libtapi --strip-components=1 -xzf $(echo $srcs | awk '{print $2}')

    # libtapi guards Apple-only build paths with NOT APPLE; we're on
    # Linux but want the Apple paths, so neutralise the guard.
    sed -i 's|NOT APPLE|FALSE|' src/projects/libtapi/CMakeLists.txt
    # ArchitectureSupport.h uses std::numeric_limits without including
    # <limits>; newer libstdc++ headers don't transitively pull it in.
    sed -i '1i #include <limits>' src/projects/libtapi/include/tapi/Core/ArchitectureSupport.h
  '';

  configurePhase = ''
    mkdir -p build && cd build
    cmake -GNinja $NIX_BUILD_TOP/src \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DLLVM_INCLUDE_TESTS=OFF
  '';

  buildPhase = ''
    ninja libtapi
  '';

  installPhase = ''
    mkdir -p $out/lib $out/include
    cp lib/libtapi* $out/lib/
    cp -ar $NIX_BUILD_TOP/src/projects/libtapi/include/tapi $out/include/
    cp -a projects/libtapi/include/tapi/Version.inc $out/include/tapi/
  '';
}
