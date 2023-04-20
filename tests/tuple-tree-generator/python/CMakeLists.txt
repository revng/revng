#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_test(NAME tuple-tree-generator-python-test-multiple-versions COMMAND
               "${CMAKE_CURRENT_SOURCE_DIR}/test.sh" "${CMAKE_SOURCE_DIR}")
set_tests_properties(tuple-tree-generator-python-test-multiple-versions
                     PROPERTIES LABELS "unit")
