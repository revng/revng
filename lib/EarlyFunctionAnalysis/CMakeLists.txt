#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Define revngEarlyFunctionAnalysis library
revng_add_analyses_library_internal(
  revngEarlyFunctionAnalysis
  AAWriterPass.cpp
  AnalyzeRegisterUsage.cpp
  AttachDebugInfo.cpp
  CFGAnalyzer.cpp
  CollectCFG.cpp
  CollectFunctionsFromCalleesPass.cpp
  CollectFunctionsFromUnusedAddressesPass.cpp
  DetectABI.cpp
  ControlFlowGraph.cpp
  FunctionSummaryOracle.cpp
  IndirectBranchInfoPrinterPass.cpp
  ControlFlowGraphCache.cpp
  Outliner.cpp
  PromoteGlobalToLocalVars.cpp
  SegregateDirectStackAccesses.cpp
  ${GENERATED_IMPLS})

llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  Analysis
  TransformUtils
  ScalarOpts
  InstCombine
  Passes
  CodeGen)

target_link_libraries(
  revngEarlyFunctionAnalysis
  revngBasicAnalyses
  revngABI
  revngSupport
  revngModel
  revngPipes
  ${LLVM_LIBRARIES})

# Header files containing efa YAML must be added here
set(EFA_HEADERS_DIR "${CMAKE_SOURCE_DIR}/include/revng/EarlyFunctionAnalysis")
set(OUTPUT_DIR
    "${CMAKE_BINARY_DIR}/include/revng/EarlyFunctionAnalysis/Generated")
set(SCHEMA_PATH
    "${CMAKE_SOURCE_DIR}/include/revng/EarlyFunctionAnalysis/efa-schema.yml")
target_tuple_tree_generator(
  revngEarlyFunctionAnalysis
  INSTALL
  HEADER_DIRECTORY
  EarlyFunctionAnalysis
  NAMESPACE
  efa
  SCHEMA_PATH
  ${SCHEMA_PATH}
  SEPARATE_STRING_TYPES
  "MetaAddress"
  SCALAR_TYPES
  "BasicBlockID"
  "MetaAddress"
  "model::FunctionAttribute::Values")

install(DIRECTORY ${OUTPUT_DIR} DESTINATION include/revng/EarlyFunctionAnalysis)
