revng_add_analyses_library(RestructureCFGPass revngc
                           ASTNode.cpp
                           ASTTree.cpp
                           BasicBlockNode.cpp
                           DotGraphObject.cpp
                           Flattening.cpp
                           MetaRegion.cpp
                           ReachabilityPass.cpp
                           RegionCFGTree.cpp
                           RestructureCFG.cpp)

target_link_libraries(RestructureCFGPass
                      revng::revngSupport
                      ${LLVM_LIBRARIES})
