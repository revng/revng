#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_library(revngBasicAnalyses
  STATIC
  FunctionCallIdentification.cpp
  GeneratedCodeBasicInfo.cpp
  ReachingDefinitionsPass.cpp)
target_link_libraries(revngBasicAnalyses revngSupport)
install(TARGETS revngBasicAnalyses
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
