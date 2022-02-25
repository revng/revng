#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_test(
  NAME tuple-tree-generator-can-import-revng-model
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test.py"
)
set_tests_properties(
  tuple-tree-generator-can-import-revng-model
  PROPERTIES
  LABELS "unit"
)
