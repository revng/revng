revng_add_analyses_library(RestructureCFGPass revngc
                           ASTNode.cpp
                           ASTTree.cpp
                           BasicBlockNode.cpp
                           ExprNode.cpp
                           Flattening.cpp
                           MetaRegion.cpp
                           RegionCFGTree.cpp
                           RestructureCFG.cpp)

target_link_libraries(RestructureCFGPass
                      TargetFunctionOption
                      revng::revngSupport
                      ${LLVM_LIBRARIES})
