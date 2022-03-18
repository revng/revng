set(CONVERT_IDB_QA_BINARIES_TEST_NAME test-convert-idb-on-qa-binaries)
add_test(NAME "${CONVERT_IDB_QA_BINARIES_TEST_NAME}"
         COMMAND "${CMAKE_SOURCE_DIR}/tests/tools/convert-idb/run-test.sh")
set_tests_properties(
  "${CONVERT_IDB_QA_BINARIES_TEST_NAME}"
  PROPERTIES
    LABELS
    "unit"
    ENVIRONMENT
    "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH};PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python"
)
