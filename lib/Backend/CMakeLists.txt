#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcBackend revngc VariableScopeAnalysisPass.cpp)

target_link_libraries(
  revngcBackend
  revngcSupport
  revng::revngModel
  revng::revngSupport
  revngcRestructureCFGPass
  revngcTargetFunctionOption
  ${LLVM_LIBRARIES})
