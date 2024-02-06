#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcInitModelTypes revngc InitModelTypes.cpp)

target_link_libraries(
  revngcInitModelTypes
  revngcSupport
  revng::revngABI
  revng::revngEarlyFunctionAnalysis
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})
