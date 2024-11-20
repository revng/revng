#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngRemoveExtractValues
                                    RemoveExtractValuesPass.cpp)

target_link_libraries(revngRemoveExtractValues revngSupport revngTypeNames
                      revngModel revngPTML ${LLVM_LIBRARIES})
