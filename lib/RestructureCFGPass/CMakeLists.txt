#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcRestructureCFGPass
  revngc
  ASTNode.cpp
  ASTTree.cpp
  BasicBlockNode.cpp
  ExprNode.cpp
  MetaRegion.cpp
  RegionCFGTree.cpp
  RestructureCFG.cpp)

target_link_libraries(
  revngcRestructureCFGPass revng::revngModel revng::revngSupport
  revngcTargetFunctionOption ${LLVM_LIBRARIES})
