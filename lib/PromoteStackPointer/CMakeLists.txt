#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcPromoteStackPointer
  revngc
  CleanupStackSizeMarkersPass.cpp
  ComputeStackAccessesBoundsPass.cpp
  DetectStackSizePass.cpp
  InstrumentStackAccessesPass.cpp
  InjectStackSizeProbesAtCallSitesPass.cpp
  PromoteStackPointerPass.cpp
  SegregateStackAccessesPass.cpp)

target_link_libraries(revngcPromoteStackPointer
  revngcSupport
  revng::revngModel
  revng::revngSupport
  revng::revngBasicAnalyses
  ${LLVM_LIBRARIES})
