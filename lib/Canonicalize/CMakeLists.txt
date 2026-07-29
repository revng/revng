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
  SwitchToStatements.cpp)

target_link_libraries(
  revngCanonicalize
  revngFunctionIsolation
  revngSupport
  revngABI
  revngLocalVariables
  revngModel
  revngPTML
  revngLift
  ${LLVM_LIBRARIES})
