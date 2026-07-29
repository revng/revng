#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngLLMRenameAnalysis
                                    LLMRenameAnalysis.cpp)

target_link_libraries(revngLLMRenameAnalysis revngModel revngSupport revngPTML
                      revngBasicAnalyses ${LLVM_LIBRARIES})
