#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPTML SHARED Doxygen.cpp IndentedOstream.cpp
                           Tag.cpp Emitter.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)

target_link_libraries(revngPTML revngABI revngSupport revngModel
                      ${LLVM_LIBRARIES})
