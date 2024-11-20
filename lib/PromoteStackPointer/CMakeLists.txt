#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngPromoteStackPointer
  CleanupStackSizeMarkersPass.cpp
  ComputeStackAccessesBoundsPass.cpp
  DetectStackSizePass.cpp
  InstrumentStackAccessesPass.cpp
  InjectStackSizeProbesAtCallSitesPass.cpp
  PromoteStackPointerPass.cpp
  RemoveStackAlignmentPass.cpp
  SegregateStackAccessesPass.cpp)

target_link_libraries(
  revngPromoteStackPointer
  revngSupport
  revngABI
  revngBasicAnalyses
  revngEarlyFunctionAnalysis
  revngModel
  revngPipes
  ${LLVM_LIBRARIES})
