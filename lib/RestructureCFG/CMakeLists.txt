#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# We need an additional library because we link this also for unit tests
revng_add_library_internal(revngScopeGraphUtils ScopeGraphUtils.cpp)
target_link_libraries(revngScopeGraphUtils revngSupport ${LLVM_LIBRARIES})

revng_add_analyses_library_internal(
  revngRestructureCFG
  DAGifyPass.cpp
  EnforceSingleExitPass.cpp
  GenericRegionInfo.cpp
  GenericRegionPass.cpp
  InlineDivergentScopesPass.cpp
  MaterializeTrivialGotoPass.cpp
  ScopeGraphAlgorithms.cpp
  ScopeGraphLoggerPass.cpp
  SelectScopePass.cpp)

target_link_libraries(revngRestructureCFG revngScopeGraphUtils revngSupport
                      revngModel ${LLVM_LIBRARIES})
