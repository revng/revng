#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPipebox SHARED Pipebox.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(
  revngPipebox
  ${Python_LIBRARIES}
  ${LLVM_LIBRARIES}
  revngModelToHeader
  revngLift
  nanobind
  revngSupport)

target_include_directories(revngPipebox PUBLIC ${Python_INCLUDE_DIRS})
