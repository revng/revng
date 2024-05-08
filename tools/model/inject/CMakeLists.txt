#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_executable(revng-model-inject Main.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES IRReader BitWriter)
target_link_libraries(revng-model-inject revngModel ${LLVM_LIBRARIES})
