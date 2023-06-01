#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngLift
  SHARED
  CodeGenerator.cpp
  CPUStateAccessAnalysisPass.cpp
  CSVOffsets.cpp
  ExternalJumpsHandler.cpp
  InstructionTranslator.cpp
  Lift.cpp
  LiftPipe.cpp
  LinkSupportPipe.cpp
  LoadBinaryPass.cpp
  JumpTargetManager.cpp
  PTCDump.cpp
  RootAnalyzer.cpp
  ValueMaterializerPass.cpp
  VariableManager.cpp)

target_link_libraries(
  revngLift
  PUBLIC dl
         m
         revngABI
         revngBasicAnalyses
         revngModelImporterDebugInfo
         revngFunctionCallIdentification
         revngModel
         revngPipes
         revngSupport
         revngTypeShrinking
         revngEarlyFunctionAnalysis
         revngValueMaterializer
         ${LLVM_LIBRARIES})
