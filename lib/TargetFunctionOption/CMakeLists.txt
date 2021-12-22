#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcTargetFunctionOption
                           revngc
                           TargetFunctionOption.cpp)

target_link_libraries(revngcTargetFunctionOption
                      ${LLVM_LIBRARIES})
