#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngYield
  SHARED
  Assembly/DisassemblyHelper.cpp
  Assembly/LLVMDisassemblerInterface.cpp
  CallGraphs/CallGraphSlices.cpp
  ControlFlow/ConvertFromEFA.cpp
  ControlFlow/Extraction.cpp
  ControlFlow/FallthroughDetection.cpp
  ControlFlow/NodeSizeCalculation.cpp
  ControlFlow/SVG.cpp
  CrossRelations.cpp
  Dump.cpp
  Plain.cpp
  PTML.cpp
  Verify.cpp)

set(DISASSEMBLER_ARCHITECTURE_SUPPORT "")
foreach(CURRENT_ARCHITECTURE AArch64 ARM Mips SystemZ X86)
  set(DISASSEMBLER_ARCHITECTURE_SUPPORT
      ${DISASSEMBLER_ARCHITECTURE_SUPPORT}
      "${CURRENT_ARCHITECTURE}Disassembler" "${CURRENT_ARCHITECTURE}Info"
      "${CURRENT_ARCHITECTURE}Desc")
endforeach(CURRENT_ARCHITECTURE)

llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  Core
  MC
  MCDisassembler
  Support
  Target
  ${DISASSEMBLER_ARCHITECTURE_SUPPORT})

target_link_libraries(
  revngYield
  revngEarlyFunctionAnalysis
  revngModelPasses
  revngPTML
  revngSupport
  revngSugiyamaGraphLayout
  revngLift
  ${LLVM_LIBRARIES})

set(INTERNAL_ASSEMBLY_HEADERS "${CMAKE_SOURCE_DIR}/include/revng/Yield")
target_tuple_tree_generator(
  revngYield
  INSTALL
  HEADER_DIRECTORY
  "Yield"
  NAMESPACE
  "yield"
  ROOT_TYPE
  Function
  SCHEMA_PATH
  "${CMAKE_BINARY_DIR}/yield-internal-schema.yml"
  SEPARATE_STRING_TYPES
  "MetaAddress"
  SCALAR_TYPES
  "BasicBlockID"
  "MetaAddress"
  "yield::ByteContainer"
  "model::FunctionAttribute::Values"
  HEADERS
  "${INTERNAL_ASSEMBLY_HEADERS}/CallEdge.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/FunctionEdge.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/FunctionEdgeBase.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/FunctionEdgeType.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/BasicBlock.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/Function.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/Instruction.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/Tag.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/TagType.h")

set(INTERNAL_ASSEMBLY_HEADERS
    "${CMAKE_SOURCE_DIR}/include/revng/Yield/CrossRelations")
target_tuple_tree_generator(
  revngYield
  INSTALL
  HEADER_DIRECTORY
  "Yield/CrossRelations"
  NAMESPACE
  "yield::crossrelations"
  ROOT_TYPE
  CrossRelations
  SCHEMA_PATH
  "${CMAKE_BINARY_DIR}/yield-cross-relations.yml"
  SEPARATE_STRING_TYPES
  "MetaAddress"
  SCALAR_TYPES
  "BasicBlockID"
  "MetaAddress"
  "yield::ByteContainer"
  HEADERS
  "${INTERNAL_ASSEMBLY_HEADERS}/CrossRelations.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/RelationDescription.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/RelationTarget.h"
  "${INTERNAL_ASSEMBLY_HEADERS}/RelationType.h")

revng_add_analyses_library_internal(
  revngYieldPipes SHARED Pipes/AssemblyPipes.cpp Pipes/CallGraphPipes.cpp
  Pipes/CFGPipes.cpp)

target_link_libraries(revngYieldPipes revngYield revngFunctionIsolation
                      revngPipes revngSupport)
