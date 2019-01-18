#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_executable(revng-lift
  BinaryFile.cpp
  CPUStateAccessAnalysisPass.cpp
  CodeGenerator.cpp
  ExternalJumpsHandler.cpp
  InstructionTranslator.cpp
  JumpTargetManager.cpp
  Main.cpp
  NoReturnAnalysis.cpp
  OSRA.cpp
  PTCDump.cpp
  SET.cpp
  SimplifyComparisonsPass.cpp
  VariableManager.cpp)

target_link_libraries(revng-lift
  dl
  m
  revngBasicAnalyses
  revngReachingDefinitions
  revngSupport
  ${LLVM_LIBRARIES})

add_custom_command(TARGET revng-lift POST_BUILD VERBATIM
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:revng-lift>" "${CMAKE_BINARY_DIR}/")

install(TARGETS revng-lift
  RUNTIME DESTINATION bin)
