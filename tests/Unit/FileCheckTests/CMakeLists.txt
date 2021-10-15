configure_file(
  ${CMAKE_SOURCE_DIR}/tests/Unit/FileCheckTests/lit.site.cfg.py.in
  ${CMAKE_BINARY_DIR}/tests/Unit/FileCheckTests/lit.site.cfg.py
  @ONLY
)

add_test(NAME llvm-lit-tests
  COMMAND lit -a ${CMAKE_BINARY_DIR}/tests/Unit/FileCheckTests/)
