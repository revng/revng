#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# To use a cross-compiler we need an external CMake project
include(ExternalProject)

# Identify QEMU, llc and so on
include(${CMAKE_SOURCE_DIR}/tests/FindTools.cmake)

enable_testing()

# Each subdirectory can define a set of tests, but most of them need to be able
# to compile programs for one of the supported architectures. Therefore we
# create a new CMake project for each supported architecture and allow the
# various subdirectories to register programs to compile.

# Initialize some variables
foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  set(TO_COMPILE_NAMES_${ARCH} "")
  set(INSTALL_DIR_${ARCH} ${CMAKE_CURRENT_BINARY_DIR}/tests/install-${ARCH})
endforeach()

# Support macro for registering a program to compile.

# ARCH: the desired architecture
# PROGRAM_NAME: a name for the output program
# SOURCES: the set of input files
# OUTPUT: name of the variable where the path to the compile binary will be
#         stored
macro(register_for_compilation ARCH PROGRAM_NAME SOURCES FLAGS OUTPUT)
  list(APPEND TO_COMPILE_NAMES_${ARCH} "${PROGRAM_NAME}")
  set(TO_COMPILE_SOURCES_${ARCH}_${PROGRAM_NAME} "${SOURCES}")
  set(TO_COMPILE_FLAGS_${ARCH}_${PROGRAM_NAME} "${FLAGS}")
  set("${OUTPUT}" "${INSTALL_DIR_${ARCH}}/bin/${PROGRAM_NAME}")
endmacro()

# Give control to the various subdirectories
include(${CMAKE_SOURCE_DIR}/tests/Runtime/RuntimeTests.cmake)
include(${CMAKE_SOURCE_DIR}/tests/Analysis/AnalysisTests.cmake)
include(${CMAKE_SOURCE_DIR}/tests/Unit/UnitTests.cmake)

# Compile the requested programs
foreach(ARCH ${SUPPORTED_ARCHITECTURES})
  # Serialize program names
  string(REPLACE ";" ":" TMP "${TO_COMPILE_NAMES_${ARCH}}")
  set(TEST_SOURCES_ARGS "-DTESTS=${TMP}")

  # Prepare CMake parameters for subproject
  # Sadly, we can't put a list into TEST_SOURCES_ARGS, since it is a list too
  foreach(PROGRAM_NAME IN LISTS TO_COMPILE_NAMES_${ARCH})
    # Serialize file names
    string(REPLACE ";" ":" SOURCES "${TO_COMPILE_SOURCES_${ARCH}_${PROGRAM_NAME}}")
    list(APPEND TEST_SOURCES_ARGS -DTEST_SOURCES_${PROGRAM_NAME}=${SOURCES})
    list(APPEND TEST_SOURCES_ARGS "-DTEST_FLAGS_${PROGRAM_NAME}=${TO_COMPILE_FLAGS_${ARCH}_${PROGRAM_NAME}}")
  endforeach()

  string(REPLACE "-" "_" NORMALIZED_ARCH "${ARCH}")
  set(TEST_CFLAGS_${ARCH} "${TEST_CFLAGS_${ARCH}} ${TEST_CFLAGS} -D_GNU_SOURCE -DTARGET_${NORMALIZED_ARCH}")

  if("${ARCH}" STREQUAL "arm")
    set(TEST_CFLAGS_${ARCH} "${TEST_CFLAGS_${ARCH}} -Wl,-Ttext-segment=0x20000")
  endif()

  list(APPEND TEST_CFLAGS_IF_AVAILABLE_${ARCH} ${TEST_CFLAGS_IF_AVAILABLE})

  # Serialize compilation flags
  string(REPLACE ";" ":" TEST_CFLAGS_IF_AVAILABLE_${ARCH} "${TEST_CFLAGS_IF_AVAILABLE_${ARCH}}")

  # Create external project using the cross-compiler
  ExternalProject_Add(TEST_PROJECT_${ARCH}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tests/${ARCH}
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR_${ARCH}}
    -DCMAKE_C_COMPILER=${C_COMPILER_${ARCH}}
    -DCMAKE_C_FLAGS=${TEST_CFLAGS_${ARCH}}
    -DCMAKE_C_FLAGS_IF_AVAILABLE=${TEST_CFLAGS_IF_AVAILABLE_${ARCH}}
    -DLINK_LIBRARIES=${TEST_LINK_LIBRARIES_${ARCH}}
    ${TEST_SOURCES_ARGS})

  # Force reconfigure each time we call make
  ExternalProject_Add_Step(TEST_PROJECT_${ARCH} forceconfigure
    COMMAND ${CMAKE_COMMAND} -E echo "Force configure"
    DEPENDEES update
    DEPENDERS configure
    ALWAYS 1)

endforeach()
