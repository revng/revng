#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(PromoteStackPointer revngc PromoteStackPointerPass.cpp)

target_link_libraries(PromoteStackPointer
  revng::revngModel
  revng::revngSupport
  revng::revngBasicAnalyses
  ${LLVM_LIBRARIES})

