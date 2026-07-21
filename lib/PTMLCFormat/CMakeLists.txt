#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPTMLCFormat SHARED Format.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support)

target_link_libraries(revngPTMLCFormat clangFormat clangToolingCore clangBasic
                      revngSupport ${LLVM_LIBRARIES})
