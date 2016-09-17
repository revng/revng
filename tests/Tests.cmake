# Sadly, to use a cross-compiler we need an external CMake project
include(ExternalProject)

configure_file(tests/li-csv-to-ld-options li-csv-to-ld-options COPYONLY)
configure_file(tests/support.c support.c COPYONLY)

# Test definitions

set(TEST_CFLAGS "-std=c99 -static -fno-pic -fno-pie -g")
set(TESTS "calc" "function_call" "floating_point" "syscall" "global")

## calc
set(TEST_SOURCES_calc "${CMAKE_SOURCE_DIR}/tests/calc.c")

set(TEST_RUNS_calc "literal" "sum" "multiplication")
set(TEST_ARGS_calc_literal "12")
set(TEST_ARGS_calc_sum "'(+ 4 5)'")
set(TEST_ARGS_calc_multiplication "'(* 5 6)'")

## function_call
set(TEST_SOURCES_function_call "${CMAKE_SOURCE_DIR}/tests/function-call.c")

set(TEST_RUNS_function_call "default")
set(TEST_ARGS_function_call_default "nope")

## floating_point
set(TEST_SOURCES_floating_point "${CMAKE_SOURCE_DIR}/tests/floating-point.c")

set(TEST_RUNS_floating_point "default")
set(TEST_ARGS_floating_point_default "nope")

## syscall
set(TEST_SOURCES_syscall "${CMAKE_SOURCE_DIR}/tests/syscall.c")

set(TEST_RUNS_syscall "default")
set(TEST_ARGS_syscall_default "nope")

## global
set(TEST_SOURCES_global "${CMAKE_SOURCE_DIR}/tests/global.c")

set(TEST_RUNS_global "default")
set(TEST_ARGS_global_default "nope")

# Get the path to some system tools we'll need

set(LLC "${LLVM_TOOLS_BINARY_DIR}/llc")
find_program(DIFF diff)

# Check which architectures are supported, we need:
# * qemu-${ARCH}
# * A cross compiler (provided by the user)
# * libtinycode-${ARCH}.so, which must be in the search path

set(SUPPORTED_ARCHITECTURES "aarch64;alpha;arm;armeb;cris;i386;m68k;microblaze;microblazeel;mips;mips64;mips64el;mipsel;mipsn32;mipsn32el;nbd;or32;ppc;ppc64;ppc64abi32;s390x;sh4;sh4eb;sparc;sparc32plus;sparc64;unicore32;x86_64")
# We can test an architecture if we have a compiler and a libtinycode-*.so
foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  set(C_COMPILER_${ARCH} ""
    CACHE
    STRING
    "Path to the C compiler to use to build tests for ${ARCH}.")

  find_library(LIBTINYCODE_${ARCH} libtinycode-${ARCH}.so
    PATHS ${QEMU_LIB_PATH}
    NO_DEFAULT_PATH)
  find_program(QEMU_${ARCH} qemu-${ARCH})

  # Try to to autodetect the compiler looking for arch*-(musl|uclibc)*-gcc in
  # PATH
  string(REPLACE ":" ";" PATH "$ENV{PATH}")
  foreach(SEARCH_PATH IN LISTS PATH)
    if (NOT C_COMPILER_${ARCH})
      set(MUSL_TOOLCHAIN "")
      set(UCLIBC_TOOLCHAIN "")
      set(TOOLCHAIN "")

      file(GLOB MUSL_TOOLCHAIN "${SEARCH_PATH}/${ARCH}*-musl*-gcc")
      file(GLOB UCLIBC_TOOLCHAIN "${SEARCH_PATH}/${ARCH}*-uclibc*-gcc")
      if(MUSL_TOOLCHAIN)
        set(TOOLCHAIN "${MUSL_TOOLCHAIN}")
      endif()
      if(UCLIBC_TOOLCHAIN)
        set(TOOLCHAIN "${UCLIBC_TOOLCHAIN}")
      endif()

      if(TOOLCHAIN)
        set(C_COMPILER_${ARCH} "${TOOLCHAIN}")
        message("${ARCH} compiler autodetected: ${C_COMPILER_${ARCH}}")
      endif()

    endif()
  endforeach()

  # If we miss one of the required components, drop the architecture
  if(LIBTINYCODE_${ARCH} STREQUAL "LIBTINYCODE_${ARCH}-NOTFOUND"
      OR C_COMPILER_${ARCH} STREQUAL "C_COMPILER_${ARCH}-NOTFOUND"
      OR QEMU_${ARCH} STREQUAL "QEMU_${ARCH}-NOTFOUND")
    list(REMOVE_ITEM SUPPORTED_ARCHITECTURES ${ARCH})
  else()
    message("Testing enabled for ${ARCH}")
  endif()
endforeach()

# Create native executable and tests
foreach(TEST_NAME ${TESTS})
  add_executable(test-native-${TEST_NAME} ${TEST_SOURCES_${TEST_NAME}})
  set_target_properties(test-native-${TEST_NAME} PROPERTIES COMPILE_FLAGS "${TEST_CFLAGS}")
  set_target_properties(test-native-${TEST_NAME} PROPERTIES LINK_FLAGS "${TEST_CFLAGS}")

  foreach(RUN_NAME ${TEST_RUNS_${TEST_NAME}})
    add_test(NAME run-test-native-${TEST_NAME}-${RUN_NAME}
      COMMAND sh -c "$<TARGET_FILE:test-native-${TEST_NAME}> ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${CMAKE_CURRENT_BINARY_DIR}/tests/run-test-native-${TEST_NAME}-${RUN_NAME}.log")
    set_tests_properties(run-test-native-${TEST_NAME}-${RUN_NAME}
        PROPERTIES LABELS "run-test-native;${TEST_NAME};${RUN_NAME}")
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

enable_testing()
foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  set(TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tests)

  # Choose install directory for subprojects
  set(BIN ${CMAKE_CURRENT_BINARY_DIR}/tests/install-${ARCH}/bin)

  # Prepare CMake parameters for subproject
  # Sadly, we can't put a list into TEST_SOURCES_ARGS, since it is a list too
  string(REPLACE ";" ":" TEST_NAMES "${TESTS}")
  set(TEST_SOURCES_ARGS "-DTESTS=${TEST_NAMES}")
  foreach(TEST_NAME ${TESTS})
    set(SOURCES "${TEST_SOURCES_${TEST_NAME}}")
    string(REPLACE ";" ":" SOURCES "${SOURCES}")
    list(APPEND TEST_SOURCES_ARGS -DTEST_SOURCES_${TEST_NAME}=${SOURCES})
  endforeach()

  string(REPLACE "-" "_" NORMALIZED_ARCH "${ARCH}")
  set(TEST_CFLAGS_${ARCH} "${TEST_CFLAGS_${ARCH}} ${TEST_CFLAGS} -D_GNU_SOURCE -DTARGET_${NORMALIZED_ARCH}")

  # Create external project using the cross-compiler
  ExternalProject_Add(TEST_PROJECT_${ARCH}
    SOURCE_DIR ${TEST_SRC}
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tests/${ARCH}
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/tests/install-${ARCH}
    -DCMAKE_C_COMPILER=${C_COMPILER_${ARCH}}
    -DCMAKE_C_FLAGS=${TEST_CFLAGS_${ARCH}}
    -DLINK_LIBRARIES=${TEST_LINK_LIBRARIES_${ARCH}}
    ${TEST_SOURCES_ARGS})

  # Force reconfigure each time we call make
  ExternalProject_Add_Step(TEST_PROJECT_${ARCH} forceconfigure
    COMMAND ${CMAKE_COMMAND} -E echo "Force configure"
    DEPENDEES update
    DEPENDERS configure
    ALWAYS 1)

  foreach(TEST_NAME ${TESTS})
    # Test to translate the compiled binary
    add_test(NAME translate-${TEST_NAME}-${ARCH}
      COMMAND sh -c "$<TARGET_FILE:revamb> --use-sections -g ll --architecture ${ARCH} ${BIN}/${TEST_NAME} ${BIN}/${TEST_NAME}.ll")
    set_tests_properties(translate-${TEST_NAME}-${ARCH}
      PROPERTIES LABELS "translate;${TEST_NAME};${ARCH}")

    # Command-line to link support.c and the translated binaries
    compile_executable("$(${CMAKE_CURRENT_SOURCE_DIR}/tests/li-csv-to-ld-options ${BIN}/${TEST_NAME}.ll.li.csv) ${BIN}/${TEST_NAME}${CMAKE_C_OUTPUT_EXTENSION} ${TEST_SRC}/support.c -DTARGET_${NORMALIZED_ARCH} -lz -lm -lrt -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast -g -fno-pie"
      "${BIN}/${TEST_NAME}.translated"
      COMPILE_TRANSLATED)

    # Compile the translated LLVM IR
    add_test(NAME compile-translated-${TEST_NAME}-${ARCH}
      COMMAND sh -c "${LLC} -O0 -filetype=obj ${BIN}/${TEST_NAME}.ll -o ${BIN}/${TEST_NAME}${CMAKE_C_OUTPUT_EXTENSION} && ${COMPILE_TRANSLATED}")
    set_tests_properties(compile-translated-${TEST_NAME}-${ARCH}
      PROPERTIES DEPENDS translate-${TEST_NAME}-${ARCH}
                 LABELS "compile-translated;${TEST_NAME};${ARCH}")

    # For each set of arguments
    foreach(RUN_NAME ${TEST_RUNS_${TEST_NAME}})
      # Test to run the translated program
      add_test(NAME run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND sh -c "${BIN}/${TEST_NAME}.translated ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${BIN}/run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}.log")
      set_tests_properties(run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES DEPENDS compile-translated-${TEST_NAME}-${ARCH}
                   LABELS "run-translated-test;${TEST_NAME};${RUN_NAME};${ARCH}")

      # Test to run the compiled program under qemu-user
      add_test(NAME run-qemu-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND sh -c "${QEMU_${ARCH}} ${BIN}/${TEST_NAME} ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${BIN}/run-qemu-test-${TEST_NAME}-${RUN_NAME}.log")
      set_tests_properties(run-qemu-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES LABELS "run-qemu-test;${TEST_NAME};${RUN_NAME};${ARCH}")

      # Check the output of the translated binary corresponds to the qemu-user's
      # one
      add_test(NAME check-with-qemu-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND "${DIFF}" "${BIN}/run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}.log" "${BIN}/run-qemu-test-${TEST_NAME}-${RUN_NAME}.log")
      set(DEPS "")
      list(APPEND DEPS "run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      list(APPEND DEPS "run-qemu-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      set_tests_properties(check-with-qemu-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES DEPENDS "${DEPS}"
                   LABELS "check-with-qemu;${TEST_NAME};${RUN_NAME};${ARCH}")

      # Check the output of the translated binary corresponds to the native's one
      add_test(NAME check-with-native-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND "${DIFF}" "${BIN}/run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}.log" "${CMAKE_CURRENT_BINARY_DIR}/tests/run-test-native-${TEST_NAME}-${RUN_NAME}.log")
      set(DEPS "")
      list(APPEND DEPS "run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}")
      list(APPEND DEPS "run-test-native-${TEST_NAME}-${RUN_NAME}")
      set_tests_properties(check-with-native-${TEST_NAME}-${RUN_NAME}-${ARCH}
        PROPERTIES DEPENDS "${DEPS}"
                   LABELS "check-with-native;${TEST_NAME};${RUN_NAME};${ARCH}")

    endforeach()
  endforeach()

endforeach()
