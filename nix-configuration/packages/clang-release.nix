{ pkgs, stdenv, python }:
# Build clang to compile QEMU helpers
stdenv.mkDerivation {
  name = "clang-release";

  src = pkgs.fetchFromGitHub {
    owner = "revng";
    repo = "llvm-project";
    rev = "e966bb52c876de8da25b301e960f886234c78007";
    hash = "sha256-XSfCHg3SpCXq9dnJg/13Kl6kVnocVWA74iLQevu/u3A=";
  };

  nativeBuildInputs = (with pkgs; [
    cmake
    ninja
  ]) ++ [ python ];

  # compiler-rt's sanitizer_common pulls in <crypt.h>, which on
  # Nix comes from libxcrypt.
  buildInputs = [ pkgs.libxcrypt ];

  cmakeFlags = [
    "-GNinja"

    "-DLLVM_INSTALL_UTILS=ON"
    "-DLLVM_ENABLE_DUMP=ON"
    "-DLLVM_ENABLE_TERMINFO=OFF"
    "-DCMAKE_CXX_STANDARD=20"
    "-DLLVM_ENABLE_Z3_SOLVER=OFF"
    "-DLLVM_ENABLE_ZLIB=ON"
    "-DLLVM_ENABLE_LIBEDIT=ON"
    "-DLLVM_ENABLE_LIBXML2=OFF"
    "-DLLVM_ENABLE_ZSTD=OFF"

    "-DBUILD_SHARED_LIBS=ON"
    "-DLLVM_ENABLE_PROJECTS=clang;compiler-rt;clang-tools-extra;lld"
    "-DLLVM_TARGETS_TO_BUILD=X86"
    "-DCOMPILER_RT_INCLUDE_TESTS=OFF"
  ];

  # Same libc++21 sancov.cpp issue as in the `llvm` derivation.
  postPatch = ''
    sed -i 's|SpecialCaseList::createOrDie({{ClIgnorelist}},|SpecialCaseList::createOrDie({static_cast<std::string>(ClIgnorelist)},|' \
      llvm/tools/sancov/sancov.cpp
  '';

  preConfigure = "cd llvm";

}
