#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(PromoteStackPointer
  revngc
  CleanupStackSizeMarkersPass.cpp
  ComputeStackAccessesBoundsPass.cpp
  DetectStackSizePass.cpp
  InstrumentStackAccessesPass.cpp
  InjectStackSizeProbesAtCallSitesPass.cpp
  PromoteStackPointerPass.cpp
  SegregateStackAccessesPass.cpp)

target_link_libraries(PromoteStackPointer
  Support
  revng::revngModel
  revng::revngSupport
  revng::revngBasicAnalyses
  ${LLVM_LIBRARIES})
