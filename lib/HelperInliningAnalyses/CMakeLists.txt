#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngHelperInliningAnalyses DetectUninlinableHelpers.cpp
  MarkInlineHelpersUpTo.cpp)

target_link_libraries(revngHelperInliningAnalyses revngBasicAnalyses revngModel
                      revngSupport ${LLVM_LIBRARIES})
