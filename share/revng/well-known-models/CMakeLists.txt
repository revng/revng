#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

file(
  GLOB
  WELL_KNOWN_BINARIES
  "${CMAKE_INSTALL_PREFIX}/share/revng/test/tests/well-known-models/*revng-qa.compiled-with-debug*"
)

add_custom_target(well-known-binaries ALL)

foreach(WELL_KNOWN_BINARY IN LISTS WELL_KNOWN_BINARIES)

  get_filename_component(BASENAME "${WELL_KNOWN_BINARY}" NAME)
  set(MODEL_PATH "share/revng/well-known-models/${BASENAME}.yml")
  set(FULL_MODEL_PATH
      "${CMAKE_BINARY_DIR}/share/revng/well-known-models/${BASENAME}")

  add_custom_command(
    OUTPUT "${FULL_MODEL_PATH}"
    COMMAND "./bin/revng" model import binary -o "${MODEL_PATH}"
            "${WELL_KNOWN_BINARY}"
    MAIN_DEPENDENCY "${WELL_KNOWN_BINARY}"
    DEPENDS revng-all-binaries
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_custom_target("import-${BASENAME}" DEPENDS "${FULL_MODEL_PATH}")

  add_dependencies(well-known-binaries "import-${BASENAME}")

endforeach()
