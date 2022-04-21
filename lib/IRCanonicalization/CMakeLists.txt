#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcIRCanonicalization
  revngc
  AddIRSerializationMarkerPass.cpp
  ExitSSAPass.cpp
  LivenessAnalysis.cpp
  MakeModelGEPPass.cpp
  TwosComplementArithmeticNormalizationPass.cpp)

target_link_libraries(
  revngcIRCanonicalization
  revngcSupport
  revngcTargetFunctionOption
  revng::revngModel
  revng::revngSupport
  revngcInitModelTypes
  ${LLVM_LIBRARIES})
