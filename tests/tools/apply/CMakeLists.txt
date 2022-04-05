macro(
  add_diff_apply_test
  TEST_NAME
  INPUT1
  OUTPUT1
  INPUT2
  OUTPUT2
  DIFF)
  add_test(
    NAME "${TEST_NAME}"
    COMMAND
      bash -c
      "${CMAKE_BINARY_DIR}/bin/revng model import binary ${INPUT1} \
      -o ${OUTPUT1} &&
      ${CMAKE_BINARY_DIR}/bin/revng model import binary ${INPUT1} \
      -o ${OUTPUT2} &&
      ${CMAKE_BINARY_DIR}/bin/revng model diff ${OUTPUT1} ${OUTPUT2} -o ${DIFF} ;
      ${CMAKE_BINARY_DIR}/bin/revng model apply ${OUTPUT1} ${DIFF} -o ${OUTPUT1} &&
      diff -u ${OUTPUT1} ${OUTPUT2}")
  set_tests_properties(
    "${TEST_NAME}" PROPERTIES LABELS "pipeline;run" ENVIRONMENT
                              "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}")
endmacro()

function(get_all_compiled_artifacts OUTPUT_VAR)

  set(TO_RETURN "")
  macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
    if("${CATEGORY}" STREQUAL "tests_runtime")
      list(APPEND TO_RETURN ${INPUT_FILE})
    endif()
  endmacro()

  register_derived_artifact("compiled" "" "" "FILE")

  set(${OUTPUT_VAR}
      ${TO_RETURN}
      PARENT_SCOPE)
endfunction()

get_all_compiled_artifacts(ALL_COMPILED)
list(GET ALL_COMPILED 1 FIRST)
list(GET ALL_COMPILED 2 SECOND)

add_diff_apply_test(
  calc_global_apply_test ${FIRST} ${CMAKE_CURRENT_BINARY_DIR}/M1.yml ${SECOND}
  ${CMAKE_CURRENT_BINARY_DIR}/M2.yml ${CMAKE_CURRENT_BINARY_DIR}/diff.yml)
set(DEPEND_ON revng-all-binaries)
