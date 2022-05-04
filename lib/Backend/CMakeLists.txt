#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcBackend revngc DecompileFunction.cpp
                           VariableScopeAnalysis.cpp)

target_link_libraries(
  revngcBackend
  revngcBeautifyGHAST
  revngcInitModelTypes
  revngcRestructureCFG
  revngcSupport
  revngcTypeNames
  revng::revngModel
  revng::revngPipes
  revng::revngSupport
  ${LLVM_LIBRARIES})
