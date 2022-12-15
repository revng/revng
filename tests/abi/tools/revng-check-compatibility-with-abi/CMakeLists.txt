#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(revng-check-compatibility-with-abi ABIArtifactParser.cpp
                     Verify.cpp Main.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(
  revng-check-compatibility-with-abi revngABI revngEarlyFunctionAnalysis
  revngModel revngSupport ${LLVM_LIBRARIES})
