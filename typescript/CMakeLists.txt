#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

configure_file("${CMAKE_SOURCE_DIR}/typescript/package.json"
               "${CMAKE_BINARY_DIR}/node_build/package.json" COPYONLY)

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/node_build/node_modules/.package-lock.json"
  COMMAND npm install --silent
  DEPENDS "${CMAKE_BINARY_DIR}/node_build/package.json"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/node_build")

add_custom_target(
  generate-node_modules
  DEPENDS "${CMAKE_BINARY_DIR}/node_build/node_modules/.package-lock.json"
          "${CMAKE_BINARY_DIR}/node_build/package.json")

# Model package generation
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/model.ts.tgz"
         "${CMAKE_BINARY_DIR}/lib/node_modules/revng-model/dist/model.js"
  COMMAND
    "${CMAKE_CURRENT_SOURCE_DIR}/build-model.sh" "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/node_build/node_modules"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  DEPENDS "${CMAKE_BINARY_DIR}/lib/typescript/model.ts")

add_custom_target(generate-model.ts.tgz ALL
                  DEPENDS "${CMAKE_BINARY_DIR}/model.ts.tgz")

install(FILES "${CMAKE_BINARY_DIR}/model.ts.tgz"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/typescript")

install(FILES "${CMAKE_BINARY_DIR}/model.ts.tgz"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/node_cache")

install(DIRECTORY "${CMAKE_BINARY_DIR}/lib/node_modules"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
