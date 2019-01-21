set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

revng_add_analyses_library(EnforceCFGCombingPass SHARED
                           EnforceCFGCombingPass.cpp
                           BasicBlockViewAnalysis.cpp
                           )

target_link_libraries(EnforceCFGCombingPass
                      RestructureCFGPass
                      revng::revngSupport
                      ${LLVM_LIBRARIES}
                      )
