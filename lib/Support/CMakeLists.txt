#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_library(revngSupport SHARED
  Assert.cpp
  CommandLine.cpp
  Debug.cpp
  DebugHelper.cpp
  ExampleAnalysis.cpp
  IRHelpers.cpp
  Statistics.cpp)

target_include_directories(revngSupport
  INTERFACE $<INSTALL_INTERFACE:include/>)

install(TARGETS revngSupport
  EXPORT revng
  LIBRARY DESTINATION lib)
