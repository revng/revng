#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngTypeNames LLVMTypeNames.cpp
                                    ModelCBuilder.cpp)

target_link_libraries(revngTypeNames revngSupport revngPTML ${LLVM_LIBRARIES})
