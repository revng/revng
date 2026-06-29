#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngCanonicalize
  ArithmeticToGEP.cpp
  DeferAllocas.cpp
  ExitSSAPass.cpp
  ExtractValueToGEP.cpp
  FixPointerSize.cpp
  SplitStructPhis.cpp
  LoopRewriteWithCanonicalIV.cpp
  PeepholeOptimizationPass.cpp
  RemoveConstantArrayReturnsPass.cpp
  RemoveLLVMAssumeCallsPass.cpp
  SimplifySwitchPass.cpp
  SplitExponentialDataflow.cpp
  SplitOverflowIntrinsicsPass.cpp
  SwitchToStatements.cpp
  TernaryReductionPass.cpp)

target_link_libraries(
  revngCanonicalize
  revngFunctionIsolation
  revngInitModelTypes
  revngSupport
  revngABI
  revngLocalVariables
  revngModel
  revngPipes
  revngPTML
  revngLift
  ${LLVM_LIBRARIES})
