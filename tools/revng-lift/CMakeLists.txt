#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(revng-lift
  BinaryFile.cpp
  CodeGenerator.cpp
  CPUStateAccessAnalysisPass.cpp
  CSVOffsets.cpp
  ExternalJumpsHandler.cpp
  InstructionTranslator.cpp
  JumpTargetManager.cpp
  Main.cpp
  PTCDump.cpp
  ProgramCounterHandler.cpp
  VariableManager.cpp)

target_link_libraries(revng-lift
  dl
  m
  revngBasicAnalyses
  revngSupport
  revngFunctionCallIdentification
  ${LLVM_LIBRARIES})
