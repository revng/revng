#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngModelImporter
                                    ImportPrototypesFromDatabase.cpp)

target_link_libraries(
  revngModelImporter
  revngModel
  revngModelPasses
  revngPipeline
  revngPipes
  revngSupport
  ${LLVM_LIBRARIES}
  sqlite3)

add_subdirectory(Binary)
add_subdirectory(DebugInfo)
