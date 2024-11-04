#
# This file is distributed under the MIT License. See LICENSE.md for details.
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
  RemoveStackAlignmentPass.cpp
  SegregateStackAccessesPass.cpp)

target_link_libraries(
  revngcPromoteStackPointer
  revngcSupport
  revng::revngABI
  revng::revngBasicAnalyses
  revng::revngEarlyFunctionAnalysis
  revng::revngModel
  revng::revngPipes
  ${LLVM_LIBRARIES})
