#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(IsolatedFunctions revngc
  IsolatedFunctions.cpp)

target_link_libraries(Decompiler
  IsolatedFunctions
  revng::revngModel
  ${LLVM_LIBRARIES})

