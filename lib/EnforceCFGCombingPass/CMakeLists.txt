revng_add_analyses_library(EnforceCFGCombingPass revngc
                           EnforceCFGCombingPass.cpp
                           BasicBlockViewAnalysis.cpp)

target_link_libraries(EnforceCFGCombingPass
                      RestructureCFGPass
                      LivenessAnalysisPass
                      revng::revngBasicAnalyses
                      revng::revngSupport
                      ${LLVM_LIBRARIES})
