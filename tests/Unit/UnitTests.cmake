#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_policy(SET CMP0060 NEW)

set(SRC "${CMAKE_SOURCE_DIR}/tests/Unit")

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

#
# reachability_library
#
include(${SRC}/Reachability/Reachability.cmake)

#
# test_reachabilitypass
#

revng_add_private_executable(test_reachabilitypass "${SRC}/ReachabilityPass.cpp")
target_include_directories(test_reachabilitypass
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${Boost_INCLUDE_DIRS}")
target_compile_definitions(test_reachabilitypass
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_reachabilitypass
  Reachability
  revng::revngSupport
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_reachabilitypass COMMAND test_reachabilitypass)

#
# test_combingpass
#

revng_add_private_executable(test_combingpass "${SRC}/CombingPass.cpp")
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
