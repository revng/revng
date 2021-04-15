#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(IsolatedFunctions revngc
  IsolatedFunctions.cpp)

target_link_libraries(IsolatedFunctions
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})

