set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_library(LivenessAnalysis SHARED
            LivenessAnalysis.cpp
            )

target_link_libraries(LivenessAnalysis
                      ${REVNG_SUPPORT_LIBRARY}
                      ${LLVM_LIBRARIES}
                      )

install(TARGETS LivenessAnalysis
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

