#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(model-export-type-graph Main.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES IRReader)
target_link_libraries(model-export-type-graph revngModel ${LLVM_LIBRARIES})
