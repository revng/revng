#
# This file is distributed under the MIT License. See LICENSE.mit for details.
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

# Typescript package generation
function(typescript_module)
  set(options)
  set(oneValueArgs TARGET_NAME)
  set(multiValueArgs)
  cmake_parse_arguments(TYPESCRIPT_MODULE "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(TARGET_NAME "${TYPESCRIPT_MODULE_TARGET_NAME}")
  add_custom_command(
    OUTPUT
      "${CMAKE_BINARY_DIR}/${TARGET_NAME}.ts.tgz"
      "${CMAKE_BINARY_DIR}/lib/node_modules/revng-${TARGET_NAME}/dist/${TARGET_NAME}.js"
    COMMAND
      "${CMAKE_CURRENT_SOURCE_DIR}/build-tupletree.sh"
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/node_build/node_modules" "${TARGET_NAME}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS "${CMAKE_BINARY_DIR}/lib/typescript/${TARGET_NAME}.ts")

  add_custom_target("generate-${TARGET_NAME}.ts.tgz" ALL
                    DEPENDS "${CMAKE_BINARY_DIR}/${TARGET_NAME}.ts.tgz")

  install(FILES "${CMAKE_BINARY_DIR}/${TARGET_NAME}.ts.tgz"
          DESTINATION "${CMAKE_INSTALL_PREFIX}/share/typescript")

  install(FILES "${CMAKE_BINARY_DIR}/${TARGET_NAME}.ts.tgz"
          DESTINATION "${CMAKE_INSTALL_PREFIX}/share/node_cache")
endfunction()

typescript_module(TARGET_NAME model)
typescript_module(TARGET_NAME pipeline-description)

install(DIRECTORY "${CMAKE_BINARY_DIR}/lib/node_modules"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
