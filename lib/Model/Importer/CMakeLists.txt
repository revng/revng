#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngModelImporter WellKnownModels.cpp)

target_link_libraries(revngModelImporter revngModel revngPipeline
                      ${LLVM_LIBRARIES})

add_subdirectory(Binary)
add_subdirectory(DebugInfo)
