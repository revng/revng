#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_policy(SET CMP0060 NEW)

set(SRC "${CMAKE_SOURCE_DIR}/tests/Unit")

set(Boost_ADDITIONAL_VERSIONS "1.63" "1.63.0")
find_package(Boost 1.63.0 REQUIRED COMPONENTS unit_test_framework)

#
# test_reachabilitypass
#

add_executable(test_reachabilitypass "${SRC}/ReachabilityPass.cpp")
target_include_directories(test_reachabilitypass
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${Boost_INCLUDE_DIRS}")
target_compile_definitions(test_reachabilitypass
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_reachabilitypass
  ReachabilityTest
  revng::revngSupport
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${LLVM_LIBRARIES})
add_test(NAME test_reachabilitypass COMMAND test_reachabilitypass)

#
# test_combingpass
#

add_executable(test_combingpass "${SRC}/CombingPass.cpp")
target_include_directories(test_combingpass
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${Boost_INCLUDE_DIRS}")
target_compile_definitions(test_combingpass
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_combingpass
  RestructureCFGPass
  revng::revngSupport
  revng::revngUnitTestHelpers
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${LLVM_LIBRARIES})
add_test(NAME test_combingpass COMMAND test_combingpass -- "${SRC}/TestGraphs/")
