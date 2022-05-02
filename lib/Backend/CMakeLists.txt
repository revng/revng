#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcBackend revngc CBackendPass.cpp
                           DecompileFunction.cpp VariableScopeAnalysisPass.cpp)

target_link_libraries(
  revngcBackend
  revngcInitModelTypes
  revngcRestructureCFGPass
  revngcSupport
  revngcTypeNames
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})
