#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(ANALYSES "cfg" "functionsboundaries" "stack_analysis")

# For each analysis, choose a file suffix and a tool to perform the diff
set(ANALYSIS_OPT_cfg "collect-cfg")
set(ANALYSIS_OPT_OUTPUT_cfg "collect-cfg-output")
set(ANALYSIS_SUFFIX_cfg ".cfg.csv")
set(ANALYSIS_DIFF_cfg "diff -u")

set(ANALYSIS_OPT_functionsboundaries "detect-abi")
set(ANALYSIS_OPT_OUTPUT_functionsboundaries "detect-function-boundaries-output")
set(ANALYSIS_SUFFIX_functionsboundaries ".functions-boundaries.csv")
set(ANALYSIS_DIFF_functionsboundaries "diff -u")

set(ANALYSIS_OPT_stack_analysis "abi-analysis")
set(ANALYSIS_OPT_OUTPUT_stack_analysis "abi-analysis-output")
set(ANALYSIS_SUFFIX_stack_analysis ".stack-analysis.json")
set(ANALYSIS_DIFF_stack_analysis "${CMAKE_SOURCE_DIR}/scripts/compare-yaml")

set(USED_REFERENCE_FILES "")

#
# Broken tests
#

# To handle this situation we need to turn the CFG into NoFunctionCalls and make
# sure consider registers that are preserved/clobbered across each function
# call. However, this means we need to integrated the StackAnalysis in
# harvesting.
list(APPEND BROKEN_TESTS_tests_analysis jump-table-base-before-function-call)

# The following tests are broken since ABI-analysis now produces results only
# for arguments/return values in registers (not on the stack)
list(APPEND BROKEN_TESTS_tests_analysis_StackAnalysis dsaof)
list(APPEND BROKEN_TESTS_tests_analysis_StackAnalysis saofc)
list(APPEND BROKEN_TESTS_tests_analysis_StackAnalysis usaof)
list(APPEND BROKEN_TESTS_tests_analysis_StackAnalysis stack-argument-contradiction)

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  if("${CATEGORY}" MATCHES "^tests_analysis.*" AND NOT "${CONFIGURATION}" STREQUAL "aarch64")
    category_to_path("${CATEGORY_PATH}" CATEGORY_PATH)
    set(COMMAND_TO_RUN "./bin/revng" lift -g ll ${INPUT_FILE} "${OUTPUT}")
    set(DEPEND_ON revng-all-binaries)

    foreach(ANALYSIS ${ANALYSES})
      set(ANALYSIS_OUTPUT "${OUTPUT}${ANALYSIS_SUFFIX_${ANALYSIS}}")
      get_filename_component(BASENAME "${OUTPUT}" NAME_WE)

      set(REFERENCE "${CMAKE_SOURCE_DIR}/${CATEGORY_PATH}/${CONFIGURATION}/${BASENAME}${ANALYSIS_SUFFIX_${ANALYSIS}}")
      list(APPEND USED_REFERENCE_FILES "${REFERENCE}")
      if(EXISTS "${REFERENCE}" AND NOT "${BASENAME}" IN_LIST "BROKEN_TESTS_${CATEGORY}")

        set(TEST_NAME test-lifted-${CATEGORY}-${ANALYSIS}-${TARGET_NAME})
        add_test(NAME ${TEST_NAME}
          COMMAND sh -c "./bin/revng opt --${ANALYSIS_OPT_${ANALYSIS}} --${ANALYSIS_OPT_OUTPUT_${ANALYSIS}}=${ANALYSIS_OUTPUT} ${OUTPUT} --debug-log=stackanalysis -o /dev/null \
          && ${ANALYSIS_DIFF_${ANALYSIS}} ${ANALYSIS_OUTPUT} ${REFERENCE}")
        set_tests_properties(${TEST_NAME} PROPERTIES LABELS "analysis;${CATEGORY};${CONFIGURATION};${ANALYSIS}")

      endif()

    endforeach()

  endif()
endmacro()
register_derived_artifact("compiled" "lifted" ".ll" "FILE")

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  if("${CATEGORY}" MATCHES "^tests_analysis.*" AND NOT "${CONFIGURATION}" STREQUAL "aarch64")
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

# Ensure all the reference files have been used
foreach(ANALYSIS ${ANALYSES})
  file(GLOB_RECURSE REFERENCE_FILES LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/tests/analysis/**/*${ANALYSIS_SUFFIX_${ANALYSIS}}")
  foreach(REFERENCE_FILE ${REFERENCE_FILES})
    if(NOT "${REFERENCE_FILE}" IN_LIST USED_REFERENCE_FILES)
      string(REPLACE ";" "\n" USED_REFERENCE_FILES "${USED_REFERENCE_FILES}")
      message(FATAL_ERROR "The following reference file has not been used:\n${REFERENCE_FILE}\nThe follwing reference files were used:\n${USED_REFERENCE_FILES}")
    endif()
  endforeach()
endforeach()
