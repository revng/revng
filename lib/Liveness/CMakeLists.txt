revng_add_analyses_library(LivenessAnalysisPass
                           revngc
                           LivenessAnalysis.cpp
                           LivenessAnalysisPass.cpp)

target_link_libraries(LivenessAnalysisPass
                      ${REVNG_SUPPORT_LIBRARY}
                      ${LLVM_LIBRARIES})
