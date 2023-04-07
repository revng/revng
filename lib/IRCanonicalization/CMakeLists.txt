#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

add_subdirectory(MarkAssignments)

revng_add_analyses_library(
  revngcIRCanonicalization
  revngc
  DuplicateReferences.cpp
  ExitSSAPass.cpp
  FoldModelGEP.cpp
  HoistStructPhis.cpp
  LoopRewriteWithCanonicalIV.cpp
  MakeLocalVariables.cpp
  MakeModelCastPass.cpp
  MakeModelGEPPass.cpp
  RemoveRefDeref.cpp
  RemoveLLVMAssumeCallsPass.cpp
  RemoveLoadStore.cpp
  RemovePointerCasts.cpp
  TwosComplementArithmeticNormalizationPass.cpp
  OperatorPrecedenceResolutionPass.cpp
  PrettyIntFormattingPass.cpp)

target_link_libraries(
  revngcIRCanonicalization
  revngcInitModelTypes
  revngcSupport
  revng::revngABI
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})
