#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcRestructureCFG
  revngc
  ASTNode.cpp
  ASTNodeUtils.cpp
  ASTTree.cpp
  BasicBlockNode.cpp
  BeautifyGHAST.cpp
  CycleEquivalenceAnalysis.cpp
  CycleEquivalencePass.cpp
  ExprNode.cpp
  FallThroughScopeAnalysis.cpp
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

target_link_libraries(revngcRestructureCFG revngcSupport revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES})
