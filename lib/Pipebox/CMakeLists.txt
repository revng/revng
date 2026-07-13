#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPipebox SHARED Pipebox.cpp LLVMPipe.cpp
                           MLIRPipe.cpp MergeLLVMModules.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(
  revngPipebox
  ${Python3_LIBRARIES}
  ${LLVM_LIBRARIES}
  revngCanonicalize
  revngClift
  revngCliftPipes
  revngDataLayoutAnalysis
  revngEditCBodyAnalysis
  revngFunctionIsolation
  revngEditCTypeAnalysis
  revngLLMRenameAnalysis
  revngLift
  revngModelImporter
  revngPromoteStackPointer
  revngRecompile
  revngRemoveExtractValues
  revngRemoveLiftingArtifacts
  revngSegmentReferences
  revngYield
  revngYieldPipes
  nanobind
  revngSupport)

target_include_directories(revngPipebox PUBLIC ${Python3_INCLUDE_DIRS})
