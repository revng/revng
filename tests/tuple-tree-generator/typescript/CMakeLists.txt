#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  if("${CATEGORY}" MATCHES "^tests_analysis.*")
    #
    # Do round-trip model -> ts -> model to check proper roundtrip
    #
    set(ROUNDTRIP_SERIALIZATION_TEST_NAME
        test-tuple-tree-generator-typescript-model-serialization-roundtrip-${CATEGORY}-${TARGET_NAME}
    )
    add_test(
      NAME "${ROUNDTRIP_SERIALIZATION_TEST_NAME}"
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/run-typescript-roundtrip-test.sh"
              "${CMAKE_CURRENT_SOURCE_DIR}/test.js" "${INPUT_FILE}")
    set_tests_properties(
      "${ROUNDTRIP_SERIALIZATION_TEST_NAME}"
      PROPERTIES LABELS "unit" ENVIRONMENT
                 "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}" WORKING_DIRECTORY
                 "${CMAKE_BINARY_DIR}")
  endif()
endmacro()
register_derived_artifact("abi-enforced-for-decompilation" "" "" "FILE")
