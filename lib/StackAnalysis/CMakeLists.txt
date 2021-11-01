#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_subdirectory(ABIAnalyses)

revng_add_analyses_library_internal(revngStackAnalysis
  AAWriterPass.cpp
  ABI.cpp
  ABIDetectionPass.cpp
  IndirectBranchInfoPrinterPass.cpp
  PromoteGlobalToLocalVars.cpp
  SegregateDirectStackAccesses.cpp
  StackAnalysis.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Analysis)

target_link_libraries(revngStackAnalysis
  revngBasicAnalyses
  revngABIAnalyses
  revngSupport
  revngModel
  ${LLVM_LIBRARIES})
