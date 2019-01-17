#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_library(revngReachingDefinitions
  STATIC
  ReachingDefinitionsPass.cpp)
target_link_libraries(revngReachingDefinitions
  revngFunctionCallIdentification
  revngStackAnalysis
  revngSupport)
install(TARGETS revngReachingDefinitions
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
