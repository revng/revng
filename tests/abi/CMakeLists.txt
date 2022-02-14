#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_subdirectory(tools/revng-abi-verify)

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  set(INPUT_FILE "${INPUT_FILE}")
  list(GET INPUT_FILE 0 COMPILED_INPUT)
  list(GET INPUT_FILE 1 COMPILED_RUN_INPUT)

  if("${CATEGORY}" MATCHES "^reference_abi_binary_(.+)" AND
      NOT "${CONFIGURATION}" STREQUAL "static_native")

    # CMAKE_MATCH_1 comes from the MATCHES before
    set(ABI_NAME "${CMAKE_MATCH_1}")

    # We use foreach but there will be only one run
    foreach(RUN IN LISTS ARTIFACT_RUNS_${ARTIFACT_CATEGORY}__${ARTIFACT})
      set(TEST_NAME runtime-abi-test-${CATEGORY}-${TARGET_NAME}-${RUN})
      set(TEMPORARY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}")
      make_directory("${TEMPORARY_DIRECTORY}")

      get_filename_component(RUNTIME_ABI_ANALYSIS_ARTIFACT "${COMPILED_RUN_INPUT}" DIRECTORY)
      set(RUNTIME_ABI_ANALYSIS_ARTIFACT "${RUNTIME_ABI_ANALYSIS_ARTIFACT}/analyzed_binary/default.stdout")

      add_test(NAME "${TEST_NAME}"
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/test.sh"
          "${ABI_NAME}"
          "${RUNTIME_ABI_ANALYSIS_ARTIFACT}"
          "${COMPILED_INPUT}"
        WORKING_DIRECTORY "${TEMPORARY_DIRECTORY}")
      set_tests_properties("${TEST_NAME}"
        PROPERTIES
        LABELS "abi;${CATEGORY};${TARGET_NAME};${CONFIGURATION}"
        ENVIRONMENT "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}")
    endforeach()
  endif()
endmacro()
register_derived_artifact("compiled;compiled-run" "" "" "FILE")
