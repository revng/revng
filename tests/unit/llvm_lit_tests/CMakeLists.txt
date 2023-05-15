configure_file(
  ${CMAKE_SOURCE_DIR}/tests/unit/llvm_lit_tests/lit.site.cfg.py.in
  ${CMAKE_BINARY_DIR}/tests/unit/llvm_lit_tests/lit.site.cfg.py @ONLY)

add_test(NAME llvm-lit-tests
         COMMAND lit -a ${CMAKE_BINARY_DIR}/tests/unit/llvm_lit_tests/)
