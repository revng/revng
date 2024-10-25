#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library(revngcRemoveExtractValues revngc
                           RemoveExtractValuesPass.cpp)

target_link_libraries(revngcRemoveExtractValues revngcSupport revngcTypeNames
                      revng::revngModel revng::revngPTML ${LLVM_LIBRARIES})
