#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_subdirectory(${CMAKE_SOURCE_DIR}/tests/abi/tools/revng-abi-verify)

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  set(INPUT_FILE "${INPUT_FILE}")
  list(GET INPUT_FILE 0 COMPILED_INPUT)
  list(GET INPUT_FILE 1 COMPILED_RUN_INPUT)

  if("${CATEGORY}" MATCHES "^reference_abi_binary_(.+)" AND 
     NOT "${CONFIGURATION}" STREQUAL "static_native")
    set(COMMAND_TO_RUN "./bin/revng" translate -i ${COMPILED_INPUT} -o "${OUTPUT}")
    set(DEPEND_ON revng-all-binaries)

    if(NOT "${CONFIGURATION}" STREQUAL "aarch64")
      foreach(RUN IN LISTS ARTIFACT_RUNS_${ARTIFACT_CATEGORY}__${ARTIFACT})

        set(OUTPUT_RUN "${OUTPUT}-${RUN}.stdout")
        set(TEST_NAME abi-reference-binary-verification-${CMAKE_MATCH_1}-${RUN})
        add_test(NAME ${TEST_NAME}
          COMMAND sh -c "${OUTPUT} ${ARTIFACT_RUNS_${ARTIFACT_CATEGORY}__${ARTIFACT}__${RUN}} > ${OUTPUT_RUN} && \
                         diff -u ${COMPILED_RUN_INPUT}/${RUN}.stdout ${OUTPUT_RUN}")
        set_tests_properties("${TEST_NAME}" PROPERTIES LABELS "abi;${CMAKE_MATCH_1};${CONFIGURATION}")

      endforeach()
    endif()

  endif()
endmacro()
register_derived_artifact("compiled;compiled-run" "translated" "" "FILE")

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  set(INPUT_FILE "${INPUT_FILE}")
  list(GET INPUT_FILE 0 COMPILED_INPUT)
  list(GET INPUT_FILE 1 COMPILED_RUN_INPUT)

  if("${CATEGORY}" MATCHES "^reference_abi_binary_.+" AND 
     NOT "${CONFIGURATION}" STREQUAL "static_native")
    set(COMMAND_TO_RUN
      "./bin/revng"
      lift
      -g ll
      ${COMPILED_INPUT}
      "${OUTPUT}")
    set(DEPEND_ON revng-all-binaries)
  endif()
endmacro()
register_derived_artifact("compiled;compiled-run" "lifted" ".ll" "FILE")


macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  set(INPUT_FILE "${INPUT_FILE}")
  list(GET INPUT_FILE 0 COMPILED_RUN_INPUT)
  list(GET INPUT_FILE 1 LIFTED_INPUT)

  if("${CATEGORY}" MATCHES "^reference_abi_binary_(.+)")
    foreach(RUN IN LISTS ARTIFACT_RUNS_${ARTIFACT_CATEGORY}__${ARTIFACT})
      set(TEST_NAME "runtime-abi-test-${CMAKE_MATCH_1}-${RUN}")
      set(TEMPORARY_DIRECTORY
          "${CMAKE_BINARY_DIR}/share/revng/tests/abi/${CMAKE_MATCH_1}/${CONFIGURATION}")

      get_filename_component(ABI_ANALYSIS_ARTIFACT ${COMPILED_RUN_INPUT} DIRECTORY)
      set(ABI_ANALYSIS_ARTIFACT "${ABI_ANALYSIS_ARTIFACT}/analyzed_binary/default.stdout")

      add_test(NAME "${TEST_NAME}"
               COMMAND bash -c "${CMAKE_SOURCE_DIR}/tests/abi/scripts/test.sh   \
                                ${CMAKE_MATCH_1} ${TEMPORARY_DIRECTORY}         \
                                ${ABI_ANALYSIS_ARTIFACT} ${LIFTED_INPUT}")
      set_tests_properties("${TEST_NAME}" PROPERTIES LABELS "abi;${CMAKE_MATCH_1};${CONFIGURATION}")
    endforeach()
  endif()
endmacro()
register_derived_artifact("compiled-run;lifted" "" "" "FILE")
