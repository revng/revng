#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngLift SHARED
  BinaryFile.cpp
  CodeGenerator.cpp
  CPUStateAccessAnalysisPass.cpp
  CSVOffsets.cpp
  ExternalJumpsHandler.cpp
  InstructionTranslator.cpp
  JumpTargetManager.cpp
  PTCDump.cpp
  VariableManager.cpp)

target_link_libraries(revngLift
  PUBLIC
  dl
  m
  revngABI
  revngBasicAnalyses
  revngDwarfImporter
  revngFunctionCallIdentification
  revngModel
  revngSupport
  revngTypeShrinking
  revngEarlyFunctionAnalysis
  ${LLVM_LIBRARIES})
