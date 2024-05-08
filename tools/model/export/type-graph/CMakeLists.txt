#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_executable(revng-model-export-type-graph Main.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES IRReader)
target_link_libraries(revng-model-export-type-graph revngModel
                      ${LLVM_LIBRARIES})
