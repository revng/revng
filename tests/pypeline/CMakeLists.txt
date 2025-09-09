#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_test(
  NAME
  test_pypeline
  COMMAND
  "${Python_EXECUTABLE}"
  -m
  pytest
  -p
  no:cacheprovider
  -v
  --rootdir
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_all.py")
set_tests_properties(
  test_pypeline
  PROPERTIES LABELS "pypeline" ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}")
