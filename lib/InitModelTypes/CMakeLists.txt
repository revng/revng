#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngInitModelTypes InitModelTypes.cpp)

target_link_libraries(revngInitModelTypes revngSupport revngABI
                      revngEarlyFunctionAnalysis revngModel ${LLVM_LIBRARIES})
