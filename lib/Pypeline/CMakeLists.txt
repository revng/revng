#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPypeline SHARED Registry.cpp)
llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(revngPypeline revngSupport ${LLVM_LIBRARIES})
target_include_directories(revngPypeline PUBLIC ${Python_INCLUDE_DIRS})
