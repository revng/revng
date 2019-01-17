#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Test definitions
set(SRC ${CMAKE_SOURCE_DIR}/tests/Runtime)

set(TEST_CFLAGS_BASE "-std=c99 -fno-pic -fno-pie -ggdb3 -fno-stack-protector")
set(TEST_CFLAGS_STATIC "${TEST_CFLAGS_BASE} -static")
set(TEST_CFLAGS "${TEST_CFLAGS_STATIC}")
set(TEST_CFLAGS_IF_AVAILABLE "-no-pie")
set(TEST_CFLAGS_NATIVE_DYNAMIC "${TEST_CFLAGS_BASE} ${NO_PIE}")
set(TEST_CFLAGS_NATIVE_STATIC "${TEST_CFLAGS_STATIC} ${NO_PIE}")
set(TESTS "calc" "function_call" "floating_point" "syscall" "global" "printf")

## calc
set(TEST_SOURCES_calc "${SRC}/calc.c")

set(TEST_RUNS_calc "literal" "sum" "multiplication")
set(TEST_ARGS_calc_literal "12")
set(TEST_ARGS_calc_sum "'(+ 4 5)'")
set(TEST_ARGS_calc_multiplication "'(* 5 6)'")

## function_call
set(TEST_SOURCES_function_call "${SRC}/function-call.c")

set(TEST_RUNS_function_call "default")
set(TEST_ARGS_function_call_default "nope")

## floating_point
set(TEST_SOURCES_floating_point "${SRC}/floating-point.c")

set(TEST_RUNS_floating_point "default")
set(TEST_ARGS_floating_point_default "nope")

## syscall
set(TEST_SOURCES_syscall "${SRC}/syscall.c")

set(TEST_RUNS_syscall "default")
set(TEST_ARGS_syscall_default "nope")

## global
set(TEST_SOURCES_global "${SRC}/global.c")

## printf
set(TEST_SOURCES_printf "${SRC}/printf.c")

set(TEST_RUNS_printf "one" "two" "three")
set(TEST_ARGS_printf_one "one")
set(TEST_ARGS_printf_two "one two")
set(TEST_ARGS_printf_three "one two three")

set(TEST_RUNS_global "default")
set(TEST_ARGS_global_default "nope")

# Create native executable and tests
foreach(TEST_NAME ${TESTS})
  # Build the static native version
  add_executable(test-native-${TEST_NAME} ${TEST_SOURCES_${TEST_NAME}})
  set_target_properties(test-native-${TEST_NAME} PROPERTIES COMPILE_FLAGS "${TEST_CFLAGS_NATIVE_STATIC}")
  set_target_properties(test-native-${TEST_NAME} PROPERTIES LINK_FLAGS "${TEST_CFLAGS_NATIVE_STATIC}")

  # Build the dynamic native version
  add_executable(test-native-dynamic-${TEST_NAME} ${TEST_SOURCES_${TEST_NAME}})
  set_target_properties(test-native-dynamic-${TEST_NAME} PROPERTIES COMPILE_FLAGS "${TEST_CFLAGS_NATIVE_DYNAMIC}")
  set_target_properties(test-native-dynamic-${TEST_NAME} PROPERTIES LINK_FLAGS "${TEST_CFLAGS_NATIVE_DYNAMIC}")

  # Translate the dynamic native version
  add_test(NAME translate-native-dynamic-${TEST_NAME}
    COMMAND "${CMAKE_BINARY_DIR}/revng" translate $<TARGET_FILE:test-native-dynamic-${TEST_NAME}>)
  set_tests_properties(translate-native-dynamic-${TEST_NAME}
    PROPERTIES LABELS "runtime;translate-native-dynamic;${TEST_NAME}")

  foreach(RUN_NAME ${TEST_RUNS_${TEST_NAME}})
    # Test for running the native version
    add_test(NAME run-test-native-${TEST_NAME}-${RUN_NAME}
      COMMAND sh -c "$<TARGET_FILE:test-native-${TEST_NAME}> ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${CMAKE_CURRENT_BINARY_DIR}/tests/run-test-native-${TEST_NAME}-${RUN_NAME}.log")
    set_tests_properties(run-test-native-${TEST_NAME}-${RUN_NAME}
        PROPERTIES LABELS "runtime;run-test-native;${TEST_NAME};${RUN_NAME}")

    # Test for running the translated dynamic version
    add_test(NAME run-translated-test-native-dynamic-${TEST_NAME}-${RUN_NAME}
      COMMAND sh -c "$<TARGET_FILE:test-native-dynamic-${TEST_NAME}>.translated ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${CMAKE_CURRENT_BINARY_DIR}/tests/run-translated-test-native-dynamic-${TEST_NAME}-${RUN_NAME}.log")
    set_tests_properties(run-translated-test-native-dynamic-${TEST_NAME}-${RUN_NAME}
      PROPERTIES DEPENDS translate-native-dynamic-${TEST_NAME}
                 LABELS "runtime;run-translated-test-native-dynamic;${TEST_NAME};${RUN_NAME}")

    # Check the output of the translated dynamic binary corresponds to the
    # native one
    add_test(NAME check-dynamic-${TEST_NAME}-${RUN_NAME}
      COMMAND "${DIFF}" "${CMAKE_CURRENT_BINARY_DIR}/tests/run-translated-test-native-dynamic-${TEST_NAME}-${RUN_NAME}.log" "${CMAKE_CURRENT_BINARY_DIR}/tests/run-test-native-${TEST_NAME}-${RUN_NAME}.log")
    set(DEPS "")
    list(APPEND DEPS "run-translated-test-native-dynamic-${TEST_NAME}-${RUN_NAME}")
    list(APPEND DEPS "run-test-native-${TEST_NAME}-${RUN_NAME}")
    set_tests_properties(check-dynamic-${TEST_NAME}-${RUN_NAME}
      PROPERTIES DEPENDS "${DEPS}"
                 LABELS "runtime;check-dynamic;${TEST_NAME};${RUN_NAME}")
  endforeach()
endforeach()

# Helper macro to "evaluate" CMake variables such as CMAKE_C_LINK_EXECUTABLE,
# which looks like this:
# <CMAKE_C_COMPILER> <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>
#                    -o <TARGET> <LINK_LIBRARIES>
macro(parse STRING OUTPUT)
  # set(COMMAND ${CMAKE_C_LINK_EXECUTABLE})
  set(COMMAND "${STRING}")
  string(REGEX MATCH "<([^>]*)>" VARNAME "${COMMAND}")
  while(VARNAME)
    string(REPLACE "<" "" VARNAME "${VARNAME}")
    string(REPLACE ">" "" VARNAME "${VARNAME}")
    string(REPLACE "<${VARNAME}>" "${${VARNAME}}" COMMAND "${COMMAND}")
    string(REGEX MATCH "<([^>]*)>" VARNAME "${COMMAND}")
  endwhile()
  set(${OUTPUT} ${COMMAND} PARENT_SCOPE)
endmacro()

# Wrapper function for parse so we don't mess around with scopes
function(compile_executable OBJECTS TARGET OUTPUT)
  parse("${CMAKE_C_LINK_EXECUTABLE}" "${OUTPUT}")
endfunction()

foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  foreach(TEST_NAME ${TESTS})
    # Register the programs for compilation
    register_for_compilation("${ARCH}" "${TEST_NAME}" "${TEST_SOURCES_${TEST_NAME}}" "" BINARY)

    # Translate the compiled binary
    add_test(NAME translate-${TEST_NAME}-${ARCH}
      COMMAND "${CMAKE_BINARY_DIR}/revng" translate ${BINARY})
    set_tests_properties(translate-${TEST_NAME}-${ARCH}
      PROPERTIES LABELS "runtime;translate;${TEST_NAME};${ARCH}")

    # Translate the compiled binary with function isolation
    add_test(NAME translate-with-isolation-${TEST_NAME}-${ARCH}
      COMMAND sh -c "cp ${BINARY} ${BINARY}.isolated-functions && ${CMAKE_BINARY_DIR}/revng translate -i ${BINARY}.isolated-functions")
    set_tests_properties(translate-with-isolation-${TEST_NAME}-${ARCH}
      PROPERTIES LABELS "runtime;translate-with-isolation;${TEST_NAME};${ARCH}")

    # For each set of arguments
    foreach(RUN_NAME ${TEST_RUNS_${TEST_NAME}})
      # Test to run the translated program
      add_test(NAME run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND sh -c "${BINARY}.translated ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${BINARY}-run-translated-test-${RUN_NAME}-${ARCH}.log")
      set_tests_properties(run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES DEPENDS translate-${TEST_NAME}-${ARCH}
                   LABELS "runtime;run-translated-test;${TEST_NAME};${RUN_NAME};${ARCH}")

       # Test to run the translated program after function isolation pass
       add_test(NAME run-translated-isolated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
         COMMAND sh -c "${BINARY}.isolated-functions.translated ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${BINARY}-run-translated-isolated-test-${RUN_NAME}-${ARCH}.log")
       set_tests_properties(run-translated-isolated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
         PROPERTIES DEPENDS translate-with-isolation-${TEST_NAME}-${ARCH}
                    LABELS "runtime;run-translated-test;function-isolation;${TEST_NAME};${RUN_NAME}${ARCH}")

      # Check the output of the translated binary corresponds to the native's
      # one
      add_test(NAME check-with-native-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND "${DIFF}" "${BINARY}-run-translated-test-${RUN_NAME}-${ARCH}.log" "${CMAKE_CURRENT_BINARY_DIR}/tests/run-test-native-${TEST_NAME}-${RUN_NAME}.log")
      set(DEPS "")
      list(APPEND DEPS "run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      list(APPEND DEPS "run-test-native-${TEST_NAME}-${RUN_NAME}")
      set_tests_properties(check-with-native-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES DEPENDS "${DEPS}"
                   LABELS "runtime;check-with-native;${TEST_NAME};${RUN_NAME};${ARCH}")

      # Check the output of the translated and isolated binary corresponds to
      # the native's one
      add_test(NAME check-isolated-with-native-${TEST_NAME}-${RUN_NAME}-${ARCH}
      COMMAND "${DIFF}" "${BINARY}-run-translated-isolated-test-${RUN_NAME}-${ARCH}.log" "${CMAKE_CURRENT_BINARY_DIR}/tests/run-test-native-${TEST_NAME}-${RUN_NAME}.log")
      set(DEPS "")
      list(APPEND DEPS "run-translated-isolated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      list(APPEND DEPS "run-test-native-${TEST_NAME}-${RUN_NAME}")
      set_tests_properties(check-isolated-with-native-${TEST_NAME}-${RUN_NAME}-${ARCH}
       PROPERTIES DEPENDS "${DEPS}"
                  LABELS "runtime;check-with-native;function-isolation;${TEST_NAME};${RUN_NAME};${ARCH}")

      # Test to run the compiled program under qemu-user
      add_test(NAME run-qemu-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND sh -c "${QEMU_${ARCH}} ${BINARY} ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${BINARY}-run-qemu-test-${RUN_NAME}.log")
      set_tests_properties(run-qemu-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES LABELS "runtime;run-qemu-test;${TEST_NAME};${RUN_NAME};${ARCH}")

      # Check the output of the translated binary corresponds to the qemu-user's
      # one
      add_test(NAME check-with-qemu-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND "${DIFF}" "${BINARY}-run-translated-test-${RUN_NAME}-${ARCH}.log" "${BINARY}-run-qemu-test-${RUN_NAME}.log")
      set(DEPS "")
      list(APPEND DEPS "run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      list(APPEND DEPS "run-qemu-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      set_tests_properties(check-with-qemu-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES DEPENDS "${DEPS}"
                   LABELS "runtime;check-with-qemu;${TEST_NAME};${RUN_NAME};${ARCH}")

      # Check the output of the translated and isolated binary corresponds to
      # the qemu-user's one
      add_test(NAME check-isolated-with-qemu-${TEST_NAME}-${RUN_NAME}-${ARCH}
      COMMAND "${DIFF}" "${BINARY}-run-translated-isolated-test-${RUN_NAME}-${ARCH}.log" "${BINARY}-run-qemu-test-${RUN_NAME}.log")
      set(DEPS "")
      list(APPEND DEPS "run-translated-isolated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      list(APPEND DEPS "run-qemu-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      set_tests_properties(check-isolated-with-qemu-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES DEPENDS "${DEPS}"
                   LABELS "runtime;check-with-qemu;function-isolation;${TEST_NAME};${RUN_NAME};${ARCH}")

    endforeach()

  endforeach()

endforeach()
