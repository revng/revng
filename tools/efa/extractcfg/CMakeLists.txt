#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(revng-efa-extractcfg Main.cpp)

target_link_libraries(revng-efa-extractcfg revngEarlyFunctionAnalysis)

target_include_directories(
  revng-efa-extractcfg PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                               ${CMAKE_CURRENT_SOURCE_DIR})

target_tuple_tree_generator(
  revng-efa-extractcfg
  HEADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/DecoratedFunction.h"
  NAMESPACE
  revng
  SCHEMA_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/schema.yml"
  HEADERS_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/Generated"
  INCLUDE_PATH_PREFIX
  ".")
