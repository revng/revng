#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngClifter SHARED Clifter.cpp)

target_link_libraries(
  revngClifter PUBLIC revngClift revngCliftImportModel revngLocalVariables
                      revngRestructureCFG)

revng_add_analyses_library_internal(revngClifterAnalyses ClifterPipe.cpp)

target_link_libraries(revngClifterAnalyses PUBLIC revngClifter
                                                  revngCliftContainer)
