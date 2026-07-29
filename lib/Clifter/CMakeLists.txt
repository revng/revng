#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngClifter SHARED Clifter.cpp)

target_link_libraries(revngClifter revngClift revngCliftImportModel
                      revngLocalVariables revngRestructureCFG)
