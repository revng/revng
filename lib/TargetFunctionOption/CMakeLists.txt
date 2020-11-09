#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(TargetFunctionOption
                           revngc
                           TargetFunctionOption.cpp)

target_link_libraries(TargetFunctionOption
                      ${LLVM_LIBRARIES})
