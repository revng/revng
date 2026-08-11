{ pkgs, stdenv, python }:
# Wrap with toPythonModule so the derivation can be fed straight to
# python.withPackages — consumers then don't need to thread
# `${nanobind}/${python.sitePackages}` into PYTHONPATH by hand.
python.pkgs.toPythonModule (stdenv.mkDerivation {
  name = "nanobind";

  src = pkgs.fetchFromGitHub {
    owner = "revng";
    repo = "nanobind";
    fetchSubmodules = true;
    rev = "a111828dd36d1ce3c8443d2bfc74ac292169a0f3";
    hash = "sha256-sxEehWW+NdoWl+EO/uZ1CzD39Fibsw/XomkUKrFDsQA=";
  };

  # standalone/CMakeLists.txt computes PYTHON_INSTALL_PATH as a path
  # relative from $out to Python_SITELIB; in nix those live in
  # different store paths, so the result escapes $out. Hard-code
  # the install destination to live inside $out instead.
  postPatch = ''
    substituteInPlace standalone/CMakeLists.txt \
      --replace 'DESTINATION "''${CMAKE_INSTALL_PREFIX}/''${PYTHON_INSTALL_PATH}/nanobind"' \
                'DESTINATION "''${CMAKE_INSTALL_PREFIX}/${python.sitePackages}/nanobind"'
  '';

  nativeBuildInputs = (with pkgs; [
    cmake
    ninja
  ]) ++ [ python ];

  preConfigure = "cd standalone";

  cmakeFlags = [
    "-GNinja"
    "-DCMAKE_CXX_STANDARD=20"
    "-DBUILD_SHARED_LIBS=ON"
  ];

})
