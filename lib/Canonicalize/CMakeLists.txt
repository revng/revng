#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcCanonicalize
  revngc
  ExitSSAPass.cpp
  FoldModelGEP.cpp
  HoistStructPhis.cpp
  LoopRewriteWithCanonicalIV.cpp
  PrepareLLVMIRForMLIR.cpp
  MakeLocalVariables.cpp
  MakeModelCastPass.cpp
  MakeModelGEPPass.cpp
  RemoveLLVMAssumeCallsPass.cpp
  RemoveLoadStore.cpp
  RemovePointerCasts.cpp
  SplitOverflowIntrinsicsPass.cpp
  SwitchToStatements.cpp
  TernaryReductionPass.cpp
  TwosComplementArithmeticNormalizationPass.cpp
  OperatorPrecedenceResolutionPass.cpp
  PrettyIntFormattingPass.cpp)

target_link_libraries(
  revngcCanonicalize
  revngcInitModelTypes
  revngcTypeNames
  revngcSupport
  revng::revngABI
  revng::revngModel
  revng::revngPTML
  revng::revngSupport
  ${LLVM_LIBRARIES})
