revng_add_analyses_library(RestructureCFGPass revngc
                           ASTNode.cpp
                           ASTTree.cpp
                           BasicBlockNode.cpp
                           Flattening.cpp
                           MetaRegion.cpp
                           RegionCFGTree.cpp
                           RestructureCFG.cpp)

target_link_libraries(RestructureCFGPass
                      revng::revngSupport
                      ${LLVM_LIBRARIES})

add_library(CombingTest SHARED DotGraphObject.cpp ReachabilityPass.cpp)
target_link_libraries(CombingTest RestructureCFGPass ${LLVM_LIBRARIES})
