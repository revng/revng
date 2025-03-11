#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# We need an additional library because we link this also for unit tests
revng_add_library_internal(revngScopeGraphUtils ScopeGraphUtils.cpp)
target_link_libraries(revngScopeGraphUtils revngSupport ${LLVM_LIBRARIES})

revng_add_analyses_library_internal(
  revngRestructureCFG
  ASTNode.cpp
  ASTNodeUtils.cpp
  ASTTree.cpp
  BasicBlockNode.cpp
  BeautifyGHAST.cpp
  DAGifyPass.cpp
  EnforceSingleExitPass.cpp
  ExprNode.cpp
  FallThroughScopeAnalysis.cpp
  GenericRegionInfo.cpp
  GenericRegionPass.cpp
  InlineDispatcherSwitch.cpp
  MetaRegion.cpp
  PromoteCallNoReturn.cpp
  RegionCFGTree.cpp
  RemoveDeadCode.cpp
  RestructureCFG.cpp
  SimplifyCompareNode.cpp
  SimplifyDualSwitch.cpp
  SimplifyHybridNot.cpp
  SimplifyImplicitStatement.cpp)

target_link_libraries(revngRestructureCFG revngScopeGraphUtils revngSupport
                      revngModel revngPipeline ${LLVM_LIBRARIES})
