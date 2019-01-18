#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_library(revngFunctionCallIdentification
  STATIC
  FunctionCallIdentification.cpp)
target_link_libraries(revngFunctionCallIdentification
  revngSupport)
install(TARGETS revngFunctionCallIdentification
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
