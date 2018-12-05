#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_library(revngSupport STATIC
  Assert.cpp
  CommandLine.cpp
  Debug.cpp
  ExampleAnalysis.cpp
  IRHelpers.cpp
  Statistics.cpp)

install(TARGETS revngSupport
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
