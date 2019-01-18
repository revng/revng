set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_library(EnforceCFGCombingPass SHARED
            EnforceCFGCombingPass.cpp
            BasicBlockViewAnalysis.cpp
            )

target_link_libraries(EnforceCFGCombingPass
                      RestructureCFGPass
                      ${REVNG_SUPPORT_LIBRARY}
                      ${LLVM_LIBRARIES}
                      )

install(TARGETS EnforceCFGCombingPass
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
