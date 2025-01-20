#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngLocalVariables
                                    LocalVariableHelpers.cpp)

target_link_libraries(revngLocalVariables revngABI revngModel revngSupport
                      ${LLVM_LIBRARIES})
