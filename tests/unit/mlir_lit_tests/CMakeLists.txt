#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

configure_file(
  ${CMAKE_SOURCE_DIR}/tests/unit/mlir_lit_tests/lit.site.cfg.py.in
  ${CMAKE_BINARY_DIR}/tests/unit/mlir_lit_tests/lit.site.cfg.py @ONLY)

add_test(NAME mlir-lit-tests
         COMMAND lit -a ${CMAKE_BINARY_DIR}/tests/unit/mlir_lit_tests/)
