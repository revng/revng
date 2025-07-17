#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
revng_add_library_internal(revngPypelineTraceRunner SHARED TraceFile.cpp
                           Runner.cpp SavePoint.cpp)
target_link_libraries(revngPypelineTraceRunner ${LLVM_LIBRARIES} revngModel
                      revngSupport sqlite3)
