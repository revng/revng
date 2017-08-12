#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(SRC ${CMAKE_SOURCE_DIR}/tests/Analysis)

set(OUTPUT_NAMES "cfg" "noreturn" "functionsboundaries" "stack-analysis")
set(OUTPUT_SUFFIX_cfg ".cfg.csv")
set(OUTPUT_SUFFIX_noreturn ".noreturn.csv")
set(OUTPUT_SUFFIX_functionsboundaries ".functions-boundaries.csv")
set(OUTPUT_SUFFIX_stack-analysis ".stack-analysis.log")

set(TESTS_arm "memset" "switch-addls" "switch-ldrls" "switch-disjoint-ranges"
  "call" "fake-function" "indirect-call" "longjmp" "indirect-tail-call")
set(TEST_SOURCES_arm_memset "${SRC}/arm/memset.S")
set(TEST_SOURCES_arm_switch-addls "${SRC}/arm/switch-addls.S")
set(TEST_SOURCES_arm_switch-ldrls "${SRC}/arm/switch-ldrls.S")
set(TEST_SOURCES_arm_switch-disjoint-ranges "${SRC}/arm/switch-disjoint-ranges.S")
set(TEST_SOURCES_arm_call "${SRC}/arm/call.S")
set(TEST_SOURCES_arm_fake-function "${SRC}/arm/fake-function.S")
set(TEST_SOURCES_arm_indirect-call "${SRC}/arm/indirect-call.S")
set(TEST_SOURCES_arm_longjmp "${SRC}/arm/longjmp.S")
set(TEST_SOURCES_arm_indirect-tail-call "${SRC}/arm/indirect-tail-call.S")

set(TESTS_x86_64 "switch-jump-table" "try-catch-ehframe"
  "call" "fibonacci" "indirect-call" "longjmp" "indirect-tail-call")
set(TEST_SOURCES_x86_64_switch-jump-table "${SRC}/x86_64/switch-jump-table.S")
set(TEST_SOURCES_x86_64_try-catch-ehframe "${SRC}/x86_64/try-catch-ehframe.S")
set(TEST_SOURCES_x86_64_call "${SRC}/x86_64/call.S")
set(TEST_SOURCES_x86_64_fibonacci "${SRC}/x86_64/fibonacci.c")
set(TEST_FLAGS_x86_64_fibonacci "-O2 -fno-stack-protector -fomit-frame-pointer -fno-reorder-functions -fno-unwind-tables -fno-asynchronous-unwind-tables -fno-stack-check -fno-optimize-sibling-calls -fno-inline-functions -fno-inline-small-functions -fno-align-functions -fno-optimize-sibling-calls")
set(TEST_SOURCES_x86_64_indirect-call "${SRC}/x86_64/indirect-call.S")
set(TEST_SOURCES_x86_64_longjmp "${SRC}/x86_64/longjmp.S")
set(TEST_SOURCES_x86_64_indirect-tail-call "${SRC}/x86_64/indirect-tail-call.S")

set(TESTS_mips "switch-jump-table")
set(TEST_SOURCES_mips_switch-jump-table "${SRC}/mips/switch-jump-table.S")

foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  foreach(TEST_NAME ${TESTS_${ARCH}})
    register_for_compilation("${ARCH}" "${TEST_NAME}" "${TEST_SOURCES_${ARCH}_${TEST_NAME}}" "-nostdlib ${TEST_FLAGS_${ARCH}_${TEST_NAME}}" BINARY)

    # Translate the compiled binary
    add_test(NAME translate-${TEST_NAME}-${ARCH}
      COMMAND $<TARGET_FILE:revamb> --functions-boundaries --use-sections -g ll "${BINARY}" "${BINARY}.ll")
    set_tests_properties(translate-${TEST_NAME}-${ARCH}
      PROPERTIES LABELS "analysis;translate;${TEST_NAME}-${ARCH}")

    # Extract all the information in a single shot
    add_test(NAME extract-info-${TEST_NAME}-${ARCH}
      COMMAND $<TARGET_FILE:revamb-dump>
              --cfg "${BINARY}.cfg.csv"
              --noreturn "${BINARY}.noreturn.csv"
              --functions-boundaries "${BINARY}.functions-boundaries.csv"
              --stack-analysis "${BINARY}.stack-analysis.log" "${BINARY}.ll")
    set_tests_properties(extract-info-${TEST_NAME}-${ARCH}
      PROPERTIES DEPENDS translate-${TEST_NAME}-${ARCH}
                 LABELS "analysis;extract-info;${TEST_NAME}-${ARCH}")

    foreach(OUTPUT_NAME ${OUTPUT_NAMES})
      set(REFERENCE_OUTPUT "${SRC}/${ARCH}/${TEST_NAME}${OUTPUT_SUFFIX_${OUTPUT_NAME}}")
      set(OUTPUT "${BINARY}${OUTPUT_SUFFIX_${OUTPUT_NAME}}")

      if(EXISTS "${REFERENCE_OUTPUT}")
        add_test(NAME check-${TEST_NAME}-${ARCH}-${OUTPUT_NAME}
          COMMAND "${DIFF}" "${REFERENCE_OUTPUT}" "${OUTPUT}")
        set_tests_properties(check-${TEST_NAME}-${ARCH}-${OUTPUT_NAME}
          PROPERTIES DEPENDS extract-info-${TEST_NAME}-${ARCH}
                     LABELS "analysis;check-with-reference;${TEST_NAME};${ARCH};${OUTPUT_NAME};${TEST_NAME}-${ARCH}")
      endif()
    endforeach()

  endforeach()
endforeach()
