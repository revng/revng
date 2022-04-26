#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)

  if(EXISTS ${INPUT_FILE})
    if("${INPUT_FILE}" MATCHES ".*/runtime/x86_64/compiled/calc")
      execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/list-tests.sh"
                      OUTPUT_VARIABLE TEST_NAMES)
      string(REPLACE "\n" ";" TEST_LIST "${TEST_NAMES}")
      foreach(TEST_NAME ${TEST_LIST})
        add_test(NAME "daemon-test-${TEST_NAME}"
                 COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test.sh")
        set_tests_properties(
          "daemon-test-${TEST_NAME}"
          PROPERTIES
            LABELS
            "daemon"
            ENVIRONMENT
            "BINARY_FILE=${INPUT_FILE};TEST_NAME=${TEST_NAME};ROOT=${CMAKE_BINARY_DIR}"
        )
      endforeach()
    endif()
  endif()

endmacro()
register_derived_artifact("compiled" "" "" "FILE")
