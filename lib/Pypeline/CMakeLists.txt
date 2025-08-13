#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_subdirectory(TraceRunner)

revng_add_library_internal(revngPypebox SHARED RegisterAll.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(revngPypebox ${Python_LIBRARIES} ${LLVM_LIBRARIES}
                      revngModelToHeader nanobind revngSupport)

target_include_directories(revngPypebox PUBLIC ${Python_INCLUDE_DIRS})
