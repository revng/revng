#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcPromoteStackPointer
  revngc
  CleanupStackSizeMarkersPass.cpp
  ComputeStackAccessesBoundsPass.cpp
  DetectStackSizePass.cpp
  InstrumentStackAccessesPass.cpp
  InjectStackSizeProbesAtCallSitesPass.cpp
  PromoteStackPointerPass.cpp
  SegregateStackAccessesPass.cpp)

target_link_libraries(
  revngcPromoteStackPointer
  revngcSupport
  revng::revngBasicAnalyses
  revng::revngEarlyFunctionAnalysis
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})
