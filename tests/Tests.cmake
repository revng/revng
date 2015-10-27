# Sadly, to use a cross-compiler we need an external CMake project
include(ExternalProject)

# Test definitions

set(TEST_CFLAGS "-std=c99 -static")
set(TESTS "calc")

## calc
set(TEST_SOURCES_calc "${CMAKE_SOURCE_DIR}/tests/calc.c")

set(TEST_RUNS_calc "literal" "sum" "multiplication")
set(TEST_ARGS_calc_literal "12")
set(TEST_ARGS_calc_sum "'(+ 4 5)'")
set(TEST_ARGS_calc_multiplication "'(* 5 6)'")

# Get the path to some system tools we'll need

find_program(LLC llc)
find_program(DIFF diff)

# Check which architectures are supported, we need:
# * qemu-${ARCH}
# * A cross compiler (provided by the user)
# * libtinycode-${ARCH}.so, which must be in the search path

set(SUPPORTED_ARCHITECTURES "aarch64;alpha;arm;armeb;cris;i386;img;io;m68k;microblaze;microblazeel;mips;mips64;mips64el;mipsel;mipsn32;mipsn32el;nbd;or32;ppc;ppc64;ppc64abi32;s390x;sh4;sh4eb;sparc;sparc32plus;sparc64;unicore32;x86_64")

# We can test an architecture if we a compiler and a libtinycode-*.so
foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  # TODO: don't harcode gcc, switch from triple to get the compiler directly
  set(TRIPLE_${ARCH} ""
    CACHE
    STRING
    "Triple to use when looking for the ${ARCH} compiler.")

  find_library(LIBTINYCODE_${ARCH} libtinycode-${ARCH}.so)
  find_program(C_COMPILER_${ARCH} ${TRIPLE_${ARCH}}-gcc)
  find_program(QEMU_${ARCH} qemu-${ARCH})

  # If we miss one of the required components, drop the architecture
  if(${LIBTINYCODE_${ARCH}} STREQUAL LIBTINYCODE_${ARCH}-NOTFOUND
      OR ${C_COMPILER_${ARCH}} STREQUAL C_COMPILER_${ARCH}-NOTFOUND
      OR ${QEMU_${ARCH}} STREQUAL QEMU_${ARCH}-NOTFOUND)
    list(REMOVE_ITEM SUPPORTED_ARCHITECTURES ${ARCH})
  else()
    message("Testing enabled for ${ARCH}")
  endif()
endforeach()

# Create native executable and tests
foreach(TEST_NAME ${TESTS})
  add_executable(test-native-${TEST_NAME} ${TEST_SOURCES_${TEST_NAME}})
  set_target_properties(test-native-${TEST_NAME} PROPERTIES COMPILE_FLAGS "${TEST_CFLAGS}")

  foreach(RUN_NAME ${TEST_RUNS_${TEST_NAME}})
    add_test(NAME run-test-native-${TEST_NAME}-${RUN_NAME}
      COMMAND sh -c "$<TARGET_FILE:test-native-${TEST_NAME}> ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${CMAKE_CURRENT_BINARY_DIR}/tests/run-test-native-${TEST_NAME}-${RUN_NAME}.log")
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

  # Create external project using the cross-compiler
  ExternalProject_Add(TEST_PROJECT_${ARCH}
    SOURCE_DIR ${TEST_SRC}
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tests/${ARCH}
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/tests/install-${ARCH}
    -DCMAKE_C_COMPILER=${C_COMPILER_${ARCH}}
    -DCMAKE_C_FLAGS=${TEST_CFLAGS}
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
      COMMAND sh -c "$<TARGET_FILE:revamb> --offset $(${CMAKE_CURRENT_SOURCE_DIR}/tests/get-function-offset ${BIN}/${TEST_NAME} root) --architecture ${ARCH} ${BIN}/${TEST_NAME} --output ${BIN}/${TEST_NAME}.ll")

    # Command-line to link support.c and the translated binaries
    compile_executable("${BIN}/${TEST_NAME}${CMAKE_C_OUTPUT_EXTENSION} ${TEST_SRC}/support.c"
      "${BIN}/${TEST_NAME}.translated"
      COMPILE_TRANSLATED)

    # Compile the translated LLVM IR
    add_test(NAME compile-translated-${TEST_NAME}-${ARCH}
      COMMAND sh -c "${LLC} -filetype=obj ${BIN}/${TEST_NAME}.ll -o ${BIN}/${TEST_NAME}${CMAKE_C_OUTPUT_EXTENSION} && ${COMPILE_TRANSLATED}")
    set_tests_properties(compile-translated-${TEST_NAME}-${ARCH} PROPERTIES DEPENDS translate-${TEST_NAME}-${ARCH})

    # For each set of arguments
    foreach(RUN_NAME ${TEST_RUNS_${TEST_NAME}})
      # Test to run the translated program
      add_test(NAME run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND sh -c "${BIN}/${TEST_NAME}.translated ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${BIN}/run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}.log")

      # Test to run the compiled program under qemu-user
      add_test(NAME run-qemu-test-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND sh -c "${QEMU_${ARCH}} ${BIN}/${TEST_NAME} ${TEST_ARGS_${TEST_NAME}_${RUN_NAME}} > ${BIN}/run-qemu-test-${TEST_NAME}-${RUN_NAME}.log")

      # Check the output of the translated binary corresponds to the qemu-user's
      # one
      add_test(NAME check-with-qemu-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND "${DIFF}" "${BIN}/run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}.log" "${BIN}/run-qemu-test-${TEST_NAME}-${RUN_NAME}.log")

      # Check the output of the translated binary corresponds to the native's one
      add_test(NAME check-with-native-${TEST_NAME}-${RUN_NAME}-${ARCH}
        COMMAND "${DIFF}" "${BIN}/run-translated-test-${TEST_NAME}-${RUN_NAME}-${ARCH}.log" "${CMAKE_CURRENT_BINARY_DIR}/tests/run-test-native-${TEST_NAME}-${RUN_NAME}.log")
    endforeach()
  endforeach()

endforeach()
