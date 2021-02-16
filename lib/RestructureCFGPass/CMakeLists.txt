#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(RestructureCFGPass
                           revngc
                           ASTNode.cpp
                           ASTTree.cpp
                           BasicBlockNode.cpp
                           ExprNode.cpp
                           MetaRegion.cpp
                           RegionCFGTree.cpp
                           RestructureCFG.cpp)

target_link_libraries(RestructureCFGPass
                      TargetFunctionOption
                      IsolatedFunctions
                      revng::revngModel
                      revng::revngSupport
                      ${LLVM_LIBRARIES})
