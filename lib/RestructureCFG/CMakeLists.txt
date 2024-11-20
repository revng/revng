#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngRestructureCFG
  ASTNode.cpp
  ASTNodeUtils.cpp
  ASTTree.cpp
  BasicBlockNode.cpp
  BeautifyGHAST.cpp
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

target_link_libraries(revngRestructureCFG revngSupport revngModel revngPipeline
                      ${LLVM_LIBRARIES})
