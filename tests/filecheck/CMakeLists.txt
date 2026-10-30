#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Each add_filecheck_test call registers one ctest that invokes lit on a single
# test file. REVNG_BUILD_DIR is exported so lit.cfg.py can resolve %root.
function(add_filecheck_test PATH)
  add_test(
    NAME "filecheck/${PATH}"
    COMMAND
      "${CMAKE_CURRENT_SOURCE_DIR}/simplit"
      "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
      "${CMAKE_SOURCE_DIR}/tests/filecheck/${PATH}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
  set_tests_properties(
    "filecheck/${PATH}" PROPERTIES LABELS "unit;filecheck" ENVIRONMENT
                                   "REVNG_BUILD_DIR=${CMAKE_BINARY_DIR}")
endfunction()

# The tests are listed in tests.txt rather than globbed: a glob would hand CMake
# control of this directory and pick up editor backups and whatever else happens
# to sit here. check-conventions is what keeps the list honest, by failing on a
# file carrying a RUN line that is missing from it.
set(FILECHECK_TESTS_LIST "${CMAKE_CURRENT_SOURCE_DIR}/tests.txt")
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${FILECHECK_TESTS_LIST}")
file(STRINGS "${FILECHECK_TESTS_LIST}" FILECHECK_TESTS REGEX "^[^#]")

foreach(F IN LISTS FILECHECK_TESTS)
  add_filecheck_test(${F})
endforeach()
