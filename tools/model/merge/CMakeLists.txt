#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(model-merge Main.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES IRReader BitWriter)
target_link_libraries(model-merge revngModel revngModelPasses ${LLVM_LIBRARIES})
