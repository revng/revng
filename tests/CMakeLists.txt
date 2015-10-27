cmake_minimum_required(VERSION 3.0)

project(Test C)

string(REPLACE ":" ";" TESTS "${TESTS}")

foreach(TEST_NAME ${TESTS})
  string(REPLACE ":" ";" TEST_SOURCES_${TEST_NAME} "${TEST_SOURCES_${TEST_NAME}}")
  add_executable(${TEST_NAME} ${TEST_SOURCES_${TEST_NAME}})
  install(TARGETS ${TEST_NAME} RUNTIME DESTINATION bin)
endforeach()
