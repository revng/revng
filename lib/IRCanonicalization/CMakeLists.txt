#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

add_subdirectory(MarkAssignments)

revng_add_analyses_library(
  revngcIRCanonicalization
  revngc
  ExitSSAPass.cpp
  FoldModelGEP.cpp
  MakeLocalVariables.cpp
  MakeModelCastPass.cpp
  MakeModelGEPPass.cpp
  RemoveRefDeref.cpp
  RemoveLoadStore.cpp
  RemovePointerCasts.cpp
  TwosComplementArithmeticNormalizationPass.cpp
  OperatorPrecedenceResolutionPass.cpp)

target_link_libraries(
  revngcIRCanonicalization revngcInitModelTypes revngcSupport revng::revngModel
  revng::revngSupport ${LLVM_LIBRARIES})
