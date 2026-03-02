#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPipebox SHARED Pipebox.cpp LLVMPipe.cpp
                           MLIRPipe.cpp MergeLLVMModules.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(
  revngPipebox
  ${Python_LIBRARIES}
  ${LLVM_LIBRARIES}
  revngBackend
  revngCanonicalize
  revngClift
  revngCliftPipes
  revngDataLayoutAnalysis
  revngFunctionIsolation
  revngImportFromCAnalysis
  revngLLMRenameAnalysis
  revngLift
  revngModelImporter
  revngModelToHeader
  revngPromoteStackPointer
  revngRecompile
  revngRemoveExtractValues
  revngRemoveLiftingArtifacts
  revngYield
  revngYieldPipes
  nanobind
  revngSupport)

target_include_directories(revngPipebox PUBLIC ${Python_INCLUDE_DIRS})
