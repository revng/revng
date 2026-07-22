#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngPromoteStackPointer
  CleanupStackSizeMarkersPass.cpp
  ComputeStackAccessesBoundsPass.cpp
  DetectStackSize.cpp
  InstrumentStackAccessesPass.cpp
  InjectStackSizeProbesAtCallSites.cpp
  PromoteStackPointer.cpp
  RemoveStackAlignmentPass.cpp
  SegregateStackAccesses.cpp)

target_link_libraries(
  revngPromoteStackPointer
  revngSupport
  revngABI
  revngBasicAnalyses
  revngEarlyFunctionAnalysis
  revngLocalVariables
  revngModel
  revngPipes
  ${LLVM_LIBRARIES})
