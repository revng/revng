#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Note: this is just a support file to create a CMake project on the fly that
# can use different compiler (a cross-compiler). Its only purpose is to compile
# and install a series of simple programs that will be used for testing.
# What you're looking for is probably Tests.cmake.

cmake_minimum_required(VERSION 2.8)

project(Test C ASM)

string(REPLACE ":" ";" TESTS "${TESTS}")

foreach(TEST_NAME ${TESTS})
  string(REPLACE ":" ";" TEST_SOURCES_${TEST_NAME} "${TEST_SOURCES_${TEST_NAME}}")
  add_executable(${TEST_NAME} ${TEST_SOURCES_${TEST_NAME}})
  set_target_properties(${TEST_NAME} PROPERTIES LINK_FLAGS "${TEST_FLAGS_${TEST_NAME}}"
                                                COMPILE_FLAGS "${TEST_FLAGS_${TEST_NAME}}")
  target_link_libraries(${TEST_NAME} "${LINK_LIBRARIES}")
  install(TARGETS ${TEST_NAME} RUNTIME DESTINATION bin)
endforeach()
