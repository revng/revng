#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(SRC ${CMAKE_SOURCE_DIR}/tests/Analysis)

# List of analyses
set(OUTPUT_NAMES "cfg" "noreturn" "functionsboundaries" "stack-analysis")

# For each analysis, choose a file suffix and a tool to perform the diff
set(OUTPUT_OPT_cfg "collect-cfg")
set(OUTPUT_SUFFIX_cfg ".cfg.csv")
set(OUTPUT_DIFF_cfg "${DIFF}")

set(OUTPUT_OPT_noreturn "collect-noreturn")
set(OUTPUT_SUFFIX_noreturn ".noreturn.csv")
set(OUTPUT_DIFF_noreturn "${DIFF}")

set(OUTPUT_OPT_functionsboundaries "detect-function-boundaries")
set(OUTPUT_SUFFIX_functionsboundaries ".functions-boundaries.csv")
set(OUTPUT_DIFF_functionsboundaries "${DIFF}")

set(OUTPUT_OPT_stack-analysis "abi-analysis")
set(OUTPUT_SUFFIX_stack-analysis ".stack-analysis.json")
set(OUTPUT_DIFF_stack-analysis ${CMAKE_SOURCE_DIR}/scripts/compare-json.py --order)

set(TESTS_arm "memset" "switch-addls" "switch-ldrls" "switch-disjoint-ranges"
  "call" "fake-function" "fake-function-without-push" "indirect-call" "longjmp"
  "indirect-tail-call")
set(TEST_SOURCES_arm_memset "${SRC}/arm/memset.S")
set(TEST_SOURCES_arm_switch-addls "${SRC}/arm/switch-addls.S")
set(TEST_SOURCES_arm_switch-ldrls "${SRC}/arm/switch-ldrls.S")
set(TEST_SOURCES_arm_switch-disjoint-ranges "${SRC}/arm/switch-disjoint-ranges.S")
set(TEST_SOURCES_arm_call "${SRC}/arm/call.S")
set(TEST_SOURCES_arm_fake-function "${SRC}/arm/fake-function.S")
set(TEST_SOURCES_arm_fake-function-without-push "${SRC}/arm/fake-function-without-push.S")
set(TEST_SOURCES_arm_indirect-call "${SRC}/arm/indirect-call.S")
set(TEST_SOURCES_arm_longjmp "${SRC}/arm/longjmp.S")
set(TEST_SOURCES_arm_indirect-tail-call "${SRC}/arm/indirect-tail-call.S")

set(TESTS_x86_64 "switch-jump-table" "rda-in-memory" "try-catch-ehframe" "call"
  "fibonacci" "indirect-call" "longjmp" "indirect-tail-call")
set(TEST_SOURCES_x86_64_switch-jump-table "${SRC}/x86_64/switch-jump-table.S")
set(TEST_SOURCES_x86_64_rda-in-memory "${SRC}/x86_64/rda-in-memory.S")
set(TEST_SOURCES_x86_64_try-catch-ehframe "${SRC}/x86_64/try-catch-ehframe.S")
set(TEST_SOURCES_x86_64_call "${SRC}/x86_64/call.S")
set(TEST_SOURCES_x86_64_fibonacci "${SRC}/x86_64/fibonacci.c")
set(TEST_FLAGS_x86_64_fibonacci "-O2 -fno-stack-protector -fomit-frame-pointer -fno-reorder-functions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-stack-check -fno-optimize-sibling-calls -fno-inline-functions -fno-inline-small-functions -fno-align-functions -fno-optimize-sibling-calls")
set(TEST_SOURCES_x86_64_indirect-call "${SRC}/x86_64/indirect-call.S")
set(TEST_SOURCES_x86_64_longjmp "${SRC}/x86_64/longjmp.S")
set(TEST_SOURCES_x86_64_indirect-tail-call "${SRC}/x86_64/indirect-tail-call.S")

# Add StackAnalysis tests
# TODO: create macros for adding tests
set(TESTS_x86_64 ${TESTS_x86_64} always-dead-return-value dead-on-one-path
  dead-register draof drvofc raofc recursion
  sometimes-dead-return-value uraof urvofc urvof
  push-pop indirect-call-callee-saved helper return_value_to_argument)
set(TEST_SOURCES_x86_64_always-dead-return-value "${SRC}/x86_64/StackAnalysis/always-dead-return-value.S")
set(TEST_SOURCES_x86_64_dead-on-one-path "${SRC}/x86_64/StackAnalysis/dead-on-one-path.S")
set(TEST_SOURCES_x86_64_dead-register "${SRC}/x86_64/StackAnalysis/dead-register.S")
set(TEST_SOURCES_x86_64_draof "${SRC}/x86_64/StackAnalysis/draof.S")
set(TEST_SOURCES_x86_64_drvofc "${SRC}/x86_64/StackAnalysis/drvofc.S")
set(TEST_SOURCES_x86_64_raofc "${SRC}/x86_64/StackAnalysis/raofc.S")
set(TEST_SOURCES_x86_64_recursion "${SRC}/x86_64/StackAnalysis/recursion.S")
set(TEST_SOURCES_x86_64_sometimes-dead-return-value "${SRC}/x86_64/StackAnalysis/sometimes-dead-return-value.S")
set(TEST_SOURCES_x86_64_uraof "${SRC}/x86_64/StackAnalysis/uraof.S")
set(TEST_SOURCES_x86_64_urvofc "${SRC}/x86_64/StackAnalysis/urvofc.S")
set(TEST_SOURCES_x86_64_urvof "${SRC}/x86_64/StackAnalysis/urvof.S")
set(TEST_SOURCES_x86_64_push-pop "${SRC}/x86_64/StackAnalysis/push-pop.S")
set(TEST_SOURCES_x86_64_indirect-call-callee-saved "${SRC}/x86_64/StackAnalysis/indirect-call-callee-saved.S")
set(TEST_SOURCES_x86_64_helper "${SRC}/x86_64/StackAnalysis/helper.S")
set(TEST_SOURCES_x86_64_return_value_to_argument "${SRC}/x86_64/StackAnalysis/return-value-to-argument.S")

set(TESTS_mips "switch-jump-table" "switch-jump-table-stack" "jump-table-base-before-function-call")
set(TEST_SOURCES_mips_switch-jump-table "${SRC}/mips/switch-jump-table.S")
set(TEST_SOURCES_mips_switch-jump-table-stack "${SRC}/mips/switch-jump-table-stack.S")
set(TEST_SOURCES_mips_jump-table-base-before-function-call "${SRC}/mips/jump-table-base-before-function-call.S")

foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  foreach(TEST_NAME ${TESTS_${ARCH}})
    register_for_compilation("${ARCH}" "${TEST_NAME}" "${TEST_SOURCES_${ARCH}_${TEST_NAME}}" "-nostdlib ${TEST_FLAGS_${ARCH}_${TEST_NAME}}" BINARY)

    # Translate the compiled binary
    add_test(NAME translate-${TEST_NAME}-${ARCH}
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/revng" lift --use-debug-symbols -g ll "${BINARY}" "${BINARY}.ll")
    set_tests_properties(translate-${TEST_NAME}-${ARCH}
      PROPERTIES LABELS "analysis;translate;${TEST_NAME}-${ARCH}")

    # Extract all the information, one-by-one

    list(GET TEST_SOURCES_${ARCH}_${TEST_NAME} 0 FIRST_SOURCE)
    get_filename_component(SOURCE_DIRECTORY "${FIRST_SOURCE}" DIRECTORY)
    get_filename_component(SOURCE_BASENAME "${FIRST_SOURCE}" NAME_WE)
    set(SOURCE_PREFIX "${SOURCE_DIRECTORY}/${SOURCE_BASENAME}")

    foreach(OUTPUT_NAME ${OUTPUT_NAMES})

      add_test(NAME extract-info-${TEST_NAME}-${ARCH}-${OUTPUT_NAME}
        COMMAND "${CMAKE_CURRENT_BINARY_DIR}/revng"
                opt
                -o /dev/null
                --${OUTPUT_OPT_${OUTPUT_NAME}}
                "--${OUTPUT_OPT_${OUTPUT_NAME}}-output=${BINARY}${OUTPUT_SUFFIX_${OUTPUT_NAME}}"
                "${BINARY}.ll")
      set_tests_properties(extract-info-${TEST_NAME}-${ARCH}-${OUTPUT_NAME}
        PROPERTIES DEPENDS translate-${TEST_NAME}-${ARCH}
        LABELS "analysis;extract-info;${TEST_NAME};${ARCH};${OUTPUT_NAME}")

      set(REFERENCE_OUTPUT "${SOURCE_PREFIX}${OUTPUT_SUFFIX_${OUTPUT_NAME}}")
      set(OUTPUT "${BINARY}${OUTPUT_SUFFIX_${OUTPUT_NAME}}")

      if(EXISTS "${REFERENCE_OUTPUT}")
        add_test(NAME check-${TEST_NAME}-${ARCH}-${OUTPUT_NAME}
          COMMAND ${OUTPUT_DIFF_${OUTPUT_NAME}} "${REFERENCE_OUTPUT}" "${OUTPUT}")
        set_tests_properties(check-${TEST_NAME}-${ARCH}-${OUTPUT_NAME}
          PROPERTIES DEPENDS extract-info-${TEST_NAME}-${ARCH}-${OUTPUT_NAME}
                     LABELS "analysis;check-with-reference;${TEST_NAME};${ARCH};${OUTPUT_NAME};${TEST_NAME}-${ARCH};analysis-${OUTPUT_NAME}")
      endif()
    endforeach()

  endforeach()
endforeach()
