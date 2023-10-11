#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcRestructureCFG
  revngc
  ASTNode.cpp
  ASTTree.cpp
  BasicBlockNode.cpp
  BeautifyGHAST.cpp
  ExprNode.cpp
  MetaRegion.cpp
  RegionCFGTree.cpp
  RestructureCFG.cpp
  SimplifyHybridNot.cpp
  SimplifyImplicitStatement.cpp)

target_link_libraries(revngcRestructureCFG revngcSupport revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES})
