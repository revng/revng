#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_executable(revng-model-apply Main.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES IRReader BitWriter)
target_link_libraries(revng-model-apply revngModel ${LLVM_LIBRARIES})
