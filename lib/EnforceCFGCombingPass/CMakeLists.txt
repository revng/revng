revng_add_analyses_library(EnforceCFGCombingPass revngc
                           EnforceCFGCombingPass.cpp
                           BasicBlockViewAnalysis.cpp)

target_link_libraries(EnforceCFGCombingPass
                      RestructureCFGPass
                      LivenessAnalysisPass
                      revng::revngSupport
                      ${LLVM_LIBRARIES})
