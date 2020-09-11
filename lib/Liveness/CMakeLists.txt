#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(LivenessAnalysis
                           revngc
                           LivenessAnalysis.cpp
                           LivenessAnalysisPass.cpp)

target_link_libraries(LivenessAnalysis
                      revng::revngSupport
                      ${LLVM_LIBRARIES})
