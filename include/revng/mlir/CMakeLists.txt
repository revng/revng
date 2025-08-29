#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# This policy mimics the behavior set in `CMakePolicy.cmake` in the official
# LLVM repo. Can be dropped when it is dropped from there.
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# This is required by the following `AddMLIR.cmake` file
include("${LLVM_DIR}/AddLLVM.cmake")

# This is needed for the `mlir_tablegen` and `add_public_tablegen_target`
# functions
include("${LLVM_DIR}/TableGen.cmake")

# This is needed for the `add_mlir_dialect` and `add_mlir_doc` functions
include("${MLIR_DIR}/AddMLIR.cmake")

add_subdirectory(Dialect)
