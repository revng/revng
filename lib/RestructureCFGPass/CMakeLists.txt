set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

revng_add_analyses_library(RestructureCFGPass SHARED
                           ASTNode.cpp
                           ASTTree.cpp
                           BasicBlockNode.cpp
                           Flattening.cpp
                           MetaRegion.cpp
                           RegionCFGTree.cpp
                           RestructureCFG.cpp
                           )

target_link_libraries(RestructureCFGPass
                      revng::revngSupport
                      ${LLVM_LIBRARIES}
                      )
