#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(revng-abi-verify
  Verify.cpp
  Main.cpp)

target_link_libraries(revng-abi-verify
  revngEarlyFunctionAnalysis
  revngModel
  revngSupport
  ${LLVM_LIBRARIES})
