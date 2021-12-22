#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  set(INPUT_FILE "${INPUT_FILE}")
  list(GET INPUT_FILE 0 COMPILED_INPUT)
  list(GET INPUT_FILE 1 COMPILED_RUN_INPUT)

  if("${CATEGORY}" STREQUAL "tests_runtime" AND NOT "${CONFIGURATION}" STREQUAL "static_native")
    set(COMMAND_TO_RUN "./bin/revng" translate -i ${COMPILED_INPUT} -o "${OUTPUT}")
    set(DEPEND_ON revng-all-binaries)

    if(NOT "${CONFIGURATION}" STREQUAL "aarch64")
      foreach(RUN IN LISTS ARTIFACT_RUNS_${ARTIFACT_CATEGORY}__${ARTIFACT})

        set(OUTPUT_RUN "${OUTPUT}-${RUN}.stdout")
        set(TEST_NAME test-translated-${CATEGORY}-${TARGET_NAME}-${RUN})
        add_test(NAME ${TEST_NAME}
          COMMAND sh -c "${OUTPUT} ${ARTIFACT_RUNS_${ARTIFACT_CATEGORY}__${ARTIFACT}__${RUN}} > ${OUTPUT_RUN} \
          && diff -u ${COMPILED_RUN_INPUT}/${RUN}.stdout ${OUTPUT_RUN}")
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "runtime;${CATEGORY};${CONFIGURATION}")

      endforeach()
    endif()

  endif()
endmacro()
register_derived_artifact("compiled;compiled-run" "translated" "" "FILE")

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  set(INPUT_FILE "${INPUT_FILE}")
  list(GET INPUT_FILE 0 COMPILED_INPUT)
  list(GET INPUT_FILE 1 COMPILED_RUN_INPUT)

  if("${CATEGORY}" STREQUAL "tests_runtime" AND NOT "${CONFIGURATION}" STREQUAL "static_native")
    set(COMMAND_TO_RUN
      "./bin/revng"
      lift
      ${COMPILED_INPUT}
      "${OUTPUT}")
    set(DEPEND_ON revng-all-binaries)
  endif()
endmacro()
register_derived_artifact("compiled;compiled-run" "lifted" ".ll" "FILE")

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  if("${CATEGORY}" STREQUAL "tests_runtime" AND NOT "${CONFIGURATION}" STREQUAL "static_native")
    set(COMMAND_TO_RUN
      "./bin/revng"
      opt
      "${INPUT_FILE}"
      --detect-abi
      --isolate
      --enforce-abi
      --promote-csvs
      --invoke-isolated-functions
      --inline-helpers
      --promote-csvs
      --remove-exceptional-functions
      -o "${OUTPUT}")
    set(DEPEND_ON revng-all-binaries)
  endif()
endmacro()
register_derived_artifact("lifted" "abi-enforced-for-decompilation" ".bc" "FILE")

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  if("${CATEGORY}" STREQUAL "tests_runtime" AND NOT "${CONFIGURATION}" STREQUAL "static_native" AND "${TARGET_NAME}" MATCHES "calc")
    set(COMMAND_TO_RUN
      "${CMAKE_CURRENT_BINARY_DIR}/bin/revng"
      opt
      "${INPUT_FILE}"
      --detect-abi
      --isolate
      --enforce-abi
      --promote-csvs
      --invoke-isolated-functions
      --inline-helpers
      --promote-csvs
      --remove-exceptional-functions
      -o "${OUTPUT}")
    set(DEPEND_ON revng-all-binaries)
  endif()
endmacro()
register_derived_artifact("lifted" "abi-enforced-for-decompilation-torture" ".bc" "FILE")
