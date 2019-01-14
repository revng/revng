set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_library(RestructureCFGPass SHARED
            ASTNode.cpp
            ASTTree.cpp
            BasicBlockNode.cpp
            RegionCFGTree.cpp
            RestructureCFG.cpp
            )

target_link_libraries(RestructureCFGPass ${REVNG_SUPPORT_LIBRARY})

install(TARGETS RestructureCFGPass
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
