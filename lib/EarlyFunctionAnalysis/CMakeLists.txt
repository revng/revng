#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_subdirectory(ABIAnalyses)

# Define revngEarlyFunctionAnalysis library
revng_add_analyses_library_internal(
  revngEarlyFunctionAnalysis
  AAWriterPass.cpp
  AttachDebugInfo.cpp
  CFGAnalyzer.cpp
  CollectCFG.cpp
  CollectFunctionsFromCalleesPass.cpp
  CollectFunctionsFromUnusedAddressesPass.cpp
  DetectABI.cpp
  FunctionMetadata.cpp
  FunctionSummaryOracle.cpp
  IndirectBranchInfoPrinterPass.cpp
  FunctionMetadataCache.cpp
  Outliner.cpp
  PromoteGlobalToLocalVars.cpp
  SegregateDirectStackAccesses.cpp
  InterproceduralAnalysis.cpp
  ${GENERATED_IMPLS})

llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  Analysis
  TransformUtils
  ScalarOpts
  InstCombine
  Passes
  CodeGen)

add_dependencies(revngEarlyFunctionAnalysis abi-analyses-headers)
target_include_directories(revngEarlyFunctionAnalysis
                           PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/ABIAnalyses")

target_link_libraries(
  revngEarlyFunctionAnalysis
  revngBasicAnalyses
  revngABI
  revngABIAnalyses
  revngSupport
  revngModel
  revngPipes
  ${LLVM_LIBRARIES})

# Header files containing efa YAML must be added here
set(EFA_HEADERS_DIR "${CMAKE_SOURCE_DIR}/include/revng/EarlyFunctionAnalysis")
set(OUTPUT_DIR
    "${CMAKE_BINARY_DIR}/include/revng/EarlyFunctionAnalysis/Generated")
set(SCHEMA_PATH "${CMAKE_BINARY_DIR}/efa-schema.yml")
target_tuple_tree_generator(
  revngEarlyFunctionAnalysis
  INSTALL
  HEADER_DIRECTORY
  EarlyFunctionAnalysis
  NAMESPACE
  efa
  SCHEMA_PATH
  ${SCHEMA_PATH}
  ROOT_TYPE
  FunctionMetadata
  SEPARATE_STRING_TYPES
  "MetaAddress"
  SCALAR_TYPES
  "BasicBlockID"
  "MetaAddress"
  "model::FunctionAttribute::Values"
  HEADERS
  "${EFA_HEADERS_DIR}/BasicBlock.h"
  "${EFA_HEADERS_DIR}/CallEdge.h"
  "${EFA_HEADERS_DIR}/FunctionEdge.h"
  "${EFA_HEADERS_DIR}/FunctionEdgeBase.h"
  "${EFA_HEADERS_DIR}/FunctionEdgeType.h"
  "${EFA_HEADERS_DIR}/FunctionMetadata.h")

install(DIRECTORY ${OUTPUT_DIR} DESTINATION include/revng/EarlyFunctionAnalysis)
