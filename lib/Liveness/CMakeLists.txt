#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcLivenessAnalysis revngc LivenessAnalysis.cpp
                           LivenessAnalysisPass.cpp)

target_link_libraries(revngcLivenessAnalysis revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES})
