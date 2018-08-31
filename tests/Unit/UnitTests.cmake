#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_policy(SET CMP0060 NEW)

set(SRC "${CMAKE_SOURCE_DIR}/tests/Unit")

set(Boost_ADDITIONAL_VERSIONS "1.63" "1.63.0")
find_package(Boost 1.63.0 REQUIRED COMPONENTS unit_test_framework)

add_executable(test_lazysmallbitvector
  "${SRC}/lazysmallbitvector.cpp"
  "${CMAKE_SOURCE_DIR}/revng-assert.cpp")
target_include_directories(test_lazysmallbitvector
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${Boost_INCLUDE_DIRS}")
target_compile_definitions(test_lazysmallbitvector
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_lazysmallbitvector
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${LLVM_LIBRARIES})
add_test(NAME test_lazysmallbitvector COMMAND test_lazysmallbitvector)

add_executable(test_stackanalysis
  "${SRC}/stackanalysis.cpp"
  "${CMAKE_SOURCE_DIR}/debug.cpp"
  "${CMAKE_SOURCE_DIR}/statistics.cpp"
  "${CMAKE_SOURCE_DIR}/revng-assert.cpp")
target_include_directories(test_stackanalysis
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${CMAKE_SOURCE_DIR}/lib/StackAnalysis"
          "${CMAKE_BINARY_DIR}/lib/StackAnalysis"
          "${Boost_INCLUDE_DIRS}")
target_compile_definitions(test_stackanalysis
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_stackanalysis
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${LLVM_LIBRARIES})
add_test(NAME test_stackanalysis COMMAND test_stackanalysis)

add_executable(test_classsentinel
  "${SRC}/classsentinel.cpp"
  "${CMAKE_SOURCE_DIR}/debug.cpp"
  "${CMAKE_SOURCE_DIR}/revng-assert.cpp")
target_include_directories(test_classsentinel
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${Boost_INCLUDE_DIRS}")
target_compile_definitions(test_classsentinel
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_classsentinel
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${LLVM_LIBRARIES})
add_test(NAME test_classsentinel COMMAND test_classsentinel)
