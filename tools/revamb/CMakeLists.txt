#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_executable(revamb
  BinaryFile.cpp
  CPUStateAccessAnalysisPass.cpp
  CodeGenerator.cpp
  ExternalJumpsHandler.cpp
  FunctionBoundariesDetectionPass.cpp
  InstructionTranslator.cpp
  JumpTargetManager.cpp
  Main.cpp
  NoReturnAnalysis.cpp
  OSRA.cpp
  PTCDump.cpp
  SET.cpp
  SimplifyComparisonsPass.cpp
  VariableManager.cpp)

target_link_libraries(revamb
  dl
  m
  revngSupport
  revngDebugHelper
  revngBasicAnalyses
  ${LLVM_LIBRARIES})

add_custom_command(TARGET revamb POST_BUILD VERBATIM
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:revamb>" "${CMAKE_BINARY_DIR}/")

install(TARGETS revamb
  RUNTIME DESTINATION bin)
