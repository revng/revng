#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(ensure-rft-equivalence Main.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES IRReader)
target_link_libraries(ensure-rft-equivalence revngModel revngModelPasses
                      ${LLVM_LIBRARIES})
