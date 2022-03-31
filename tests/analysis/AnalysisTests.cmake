#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(USED_REFERENCE_FILES "")

#
# Broken tests
#

# To handle this situation we need to turn the CFG into NoFunctionCalls and make
# sure consider registers that are preserved/clobbered across each function
# call. However, this means we need to integrate EarlyFunctionAnalysis in
# harvesting.
list(APPEND BROKEN_TESTS_tests_analysis jump-table-base-before-function-call)

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  if("${CATEGORY}" MATCHES "^tests_analysis.*" AND NOT "${CONFIGURATION}"
                                                   STREQUAL "aarch64")
    #
    # Perform lifting
    #
    category_to_path("${CATEGORY_PATH}" CATEGORY_PATH)
    set(COMMAND_TO_RUN "./bin/revng" lift ${INPUT_FILE} "${OUTPUT}")
    set(DEPEND_ON revng-all-binaries)

    set(ACTUAL_CFG "${OUTPUT}.yml")
    get_filename_component(BASENAME "${OUTPUT}" NAME_WE)

    set(REFERENCE_MODEL
        "${CMAKE_SOURCE_DIR}/${CATEGORY_PATH}/${CONFIGURATION}/${BASENAME}.yml")
    list(APPEND USED_REFERENCE_FILES "${REFERENCE_MODEL}")
    if(EXISTS "${REFERENCE_MODEL}" AND NOT "${BASENAME}" IN_LIST
                                       "BROKEN_TESTS_${CATEGORY}")
      #
      # Run EFA --collect-cfg and compare with ground truth
      #
      set(TEST_NAME test-lifted-${CATEGORY}-${TARGET_NAME}-model)
      add_test(
        NAME ${TEST_NAME}
        COMMAND
          sh -c "./bin/revng opt ${OUTPUT} --detect-abi --collect-cfg -S \
          | ./bin/revng efa-extractcfg \
          | ./bin/revng model to-json --remap > ${ACTUAL_CFG} \
          && ./bin/revng model compare ${ACTUAL_CFG} ${REFERENCE_MODEL}")
      set_tests_properties(
        ${TEST_NAME}
        PROPERTIES LABELS
                   "model;analysis;${CATEGORY};${CONFIGURATION};${ANALYSIS}")

    endif()

  endif()
endmacro()
register_derived_artifact("compiled" "lifted" ".ll" "FILE")

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  if("${CATEGORY}" MATCHES "^tests_analysis.*" AND NOT "${CONFIGURATION}"
                                                   STREQUAL "aarch64")
    set(COMMAND_TO_RUN
        "./bin/revng"
        llvm
        pipeline
        "${INPUT_FILE}"
        "${OUTPUT}"
        Lift
        EnforceABI
        "*:CSVsPromoted")
    set(DEPEND_ON revng-all-binaries)

    #
    # Ensure we can load the model
    #
    set(TEST_NAME
        test-abi-enforced-for-decompilation-${CATEGORY}-${TARGET_NAME}-load-model
    )
    add_test(NAME ${TEST_NAME}
             COMMAND sh -c
                     "./bin/revng opt ${OUTPUT} --load-model -o /dev/null")
    set_tests_properties(
      ${TEST_NAME}
      PROPERTIES LABELS
                 "load-model;analysis;${CATEGORY};${CONFIGURATION};${ANALYSIS}")

    set(ROUNDTRIP_SERIALIZATION_TEST_NAME
        test-tuple-tree-generator-python-model-serialization-roundtrip-${CATEGORY}-${TARGET_NAME}
    )
    add_test(
      NAME "${ROUNDTRIP_SERIALIZATION_TEST_NAME}"
      COMMAND
        "${CMAKE_SOURCE_DIR}/tests/tuple-tree-generator/run-serialization-roundtrip-test.sh"
        "${OUTPUT}")
    set_tests_properties(
      "${ROUNDTRIP_SERIALIZATION_TEST_NAME}"
      PROPERTIES
        LABELS
        "unit"
        ENVIRONMENT
        "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH};PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python"
    )
  endif()
endmacro()
register_derived_artifact("lifted" "abi-enforced-for-decompilation" ".bc"
                          "FILE")

# Ensure all the reference files have been used
foreach(ANALYSIS ${ANALYSES})
  file(
    GLOB_RECURSE REFERENCE_FILES
    LIST_DIRECTORIES false
    "${CMAKE_SOURCE_DIR}/tests/analysis/**/*.yml")
  foreach(REFERENCE_FILE ${REFERENCE_FILES})
    if(NOT "${REFERENCE_FILE}" IN_LIST USED_REFERENCE_FILES)
      string(REPLACE ";" "\n" USED_REFERENCE_FILES "${USED_REFERENCE_FILES}")
      message(
        FATAL_ERROR
          "The following reference file has not been used:\n${REFERENCE_FILE}\nThe following reference files were used:\n${USED_REFERENCE_FILES}"
      )
    endif()
  endforeach()
endforeach()
