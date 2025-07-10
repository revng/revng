#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngRegistry SHARED Registry.cpp)
llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(revngRegistry revngSupport ${LLVM_LIBRARIES})
target_include_directories(revngRegistry PUBLIC ${Python_INCLUDE_DIRS})
