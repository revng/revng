#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcIRCanonicalization
  revngc
  AddIRSerializationMarkerPass.cpp
  MakeModelGEPPass.cpp
  ExitSSAPass.cpp
  TwosComplementArithmeticNormalizationPass.cpp)

target_link_libraries(revngcIRCanonicalization
  revngcDecompiler
  revngcLivenessAnalysis
  revngcMarkForSerialization
  revngcSupport
  revngcTargetFunctionOption
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})

