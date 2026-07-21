#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPTML SHARED Doxygen.cpp Tag.cpp PTMLEmitter.cpp
                           CTokenEmitter.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)

target_link_libraries(revngPTML revngABI revngSupport revngModel
                      revngPTMLCFormat ${LLVM_LIBRARIES})
