set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

# The library with the LivenessAnalysis
add_library(LivenessAnalysis SHARED
            LivenessAnalysis.cpp
            )

target_link_libraries(LivenessAnalysis
                      ${REVNG_SUPPORT_LIBRARY}
                      ${LLVM_LIBRARIES}
                      )

# The shared library with the LivenessAnalysisPass FunctionPass.
# It can be loaded by opt with the -load option to run the analysis.
add_library(LivenessAnalysisPass SHARED
            LivenessAnalysisPass.cpp
            )

target_link_libraries(LivenessAnalysisPass
                      LivenessAnalysis
                      ${LLVM_LIBRARIES}
                      )

install(TARGETS LivenessAnalysis
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(TARGETS LivenessAnalysisPass
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
