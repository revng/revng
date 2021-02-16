#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(AdjustStackPointer revngc AdjustStackPointerPass.cpp)

target_link_libraries(AdjustStackPointer
  IsolatedFunctions
  revng::revngModel
  revng::revngSupport
  revng::revngBasicAnalyses
  ${LLVM_LIBRARIES})


