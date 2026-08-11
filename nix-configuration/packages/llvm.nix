{ pkgs, ccacheStdenv, python }:
# Build our LLVM fork
ccacheStdenv.mkDerivation {
  name = "llvm";

  src = pkgs.fetchFromGitHub {
    owner = "revng";
    repo = "llvm-project";
    # HEAD of github.com/revng/llvm-project develop as tracked by
    # orchestra. Adds:
    #  - `GT` template parameter to po_ext_iterator / ipo_iterator /
    #    ReversePostOrderTraversal so callers (specifically MFP.h's
    #    use of GraphTraits<Inverse<RegionCFG<…>*>>) can drive the
    #    iterator with custom graph traits — without it, the legacy
    #    decompile path SIGABRTs at runtime in the Inverse case.
    #  - a 4th `PadKeyColumn` argument to `llvm::yaml::Output`,
    #    required by `include/revng/Support/YAMLTraits.h`.
    rev = "83b85f196975";
    hash = "sha256-lUiYXtiild1bBRJLDtxIeZ85EiD/ELaVAtq/9TSxhhw=";
  };

  nativeBuildInputs = (with pkgs; [
    cmake
    ninja
    zlib
    libedit
    zstd
  ]) ++ [ python ];

  cmakeFlags = [
    "-GNinja"

    "-DCMAKE_C_FLAGS=-O2"
    "-DCMAKE_CXX_FLAGS=-O2"
    "-DCMAKE_BUILD_TYPE=Debug"

    "-DCMAKE_INSTALL_BINDIR=libexec"

    "-DLLVM_INSTALL_UTILS=ON"
    "-DLLVM_ENABLE_DUMP=ON"
    "-DLLVM_ENABLE_TERMINFO=OFF"
    "-DCMAKE_CXX_STANDARD=20"
    "-DLLVM_ENABLE_Z3_SOLVER=OFF"
    "-DLLVM_ENABLE_ZLIB=ON"
    "-DLLVM_ENABLE_LIBEDIT=ON"
    "-DLLVM_ENABLE_LIBXML2=OFF"
    # Our llvm fork patches `IRReader.cpp` so that when the input
    # buffer is a zstd frame (magic `28 b5 2f fd`) it transparently
    # decompresses before handing to the bitcode/.ll parser. revng's
    # `revng artifact … <emit-llvm-artifact>` writes the IR through
    # zstd to keep the pipeline-cache containers compact, so a
    # downstream `revng opt -S` sees a zstd stream rather than raw
    # IR. With ZSTD off the patch is dead code, IRReader feeds the
    # raw zstd bytes to the .ll parser, and opt dies with "expected
    # top-level entity" on tests like
    # `SegregateStackAccesses/dynamic_native/.../filecheck`.
    "-DLLVM_ENABLE_ZSTD=ON"

    "-DBUILD_SHARED_LIBS=ON"
    "-DLLVM_ENABLE_PROJECTS=clang;mlir;lld"
    "-DLLVM_TARGETS_TO_BUILD=AArch64;ARM;Mips;SystemZ;X86"
    "-DCMAKE_CXX_FLAGS=-Wno-global-constructors"
  ];

  # sancov.cpp uses `{{ClIgnorelist}}` to build a vector<string>;
  # the inner brace tries to copy-construct std::string from a
  # cl::opt<std::string>, which fails under libc++21 because the
  # templated basic_string(const _Tp&) ctor is now `explicit`.
  # Force a direct-init conversion via static_cast.
  postPatch = ''
    sed -i 's|SpecialCaseList::createOrDie({{ClIgnorelist}},|SpecialCaseList::createOrDie({static_cast<std::string>(ClIgnorelist)},|' \
      llvm/tools/sancov/sancov.cpp
  '';

  preConfigure = "cd llvm";

}
