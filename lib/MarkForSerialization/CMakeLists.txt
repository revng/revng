#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcMarkForSerialization revngc
                           MarkForSerializationPass.cpp)

target_link_libraries(
  revngcMarkForSerialization
  revngcLivenessAnalysis
  revngcRestructureCFGPass
  revngcSupport
  revngcTargetFunctionOption
  revng::revngSupport
  ${LLVM_LIBRARIES})
