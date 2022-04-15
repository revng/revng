#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

configure_file("${CMAKE_SOURCE_DIR}/typescript/package.json"
               "${CMAKE_BINARY_DIR}/node_build/package.json" COPYONLY)

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/node_build/node_modules/.yarn-integrity"
  COMMAND yarn -s install
  DEPENDS "${CMAKE_BINARY_DIR}/node_build/package.json"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/node_build")

add_custom_target(
  generate-node_modules
  DEPENDS "${CMAKE_BINARY_DIR}/node_build/node_modules/.yarn-integrity")
