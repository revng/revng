#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_subdirectory(python)
add_subdirectory(cpp)

revng_add_test(
  NAME two-stage-model-edit COMMAND
  "${CMAKE_CURRENT_SOURCE_DIR}/two-stage-model-edit.sh" "${CMAKE_BINARY_DIR}")
set_tests_properties(
  two-stage-model-edit
  PROPERTIES LABELS "pypeline;test" ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}")
