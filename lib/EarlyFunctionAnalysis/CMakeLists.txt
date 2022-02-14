#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_subdirectory(ABIAnalyses)

revng_add_analyses_library_internal(revngEarlyFunctionAnalysis
  AAWriterPass.cpp
  ABIDetectionPass.cpp
  IndirectBranchInfoPrinterPass.cpp
  PromoteGlobalToLocalVars.cpp
  SegregateDirectStackAccesses.cpp
  EarlyFunctionAnalysis.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Analysis)

target_link_libraries(revngEarlyFunctionAnalysis
  revngBasicAnalyses
  revngABIAnalyses
  revngSupport
  revngModel
  ${LLVM_LIBRARIES})
