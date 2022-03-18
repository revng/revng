#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

cmake_policy(SET CMP0060 NEW)

include(${CMAKE_INSTALL_PREFIX}/share/revng/qa/cmake/revng-qa.cmake)

set(SRC "${CMAKE_SOURCE_DIR}/tests/Unit")

include(${SRC}/llvm-lit-tests/CMakeLists.txt)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

#
# reachability_library
#
include(${SRC}/Reachability/Reachability.cmake)

#
# test_reachabilitypass
#

revng_add_test_executable(test_reachabilitypass "${SRC}/ReachabilityPass.cpp")
target_compile_definitions(test_reachabilitypass
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(
  test_reachabilitypass PRIVATE "${CMAKE_SOURCE_DIR}" "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_reachabilitypass revngcReachability revng::revngModel
  revng::revngSupport Boost::unit_test_framework ${LLVM_LIBRARIES})
add_test(NAME test_reachabilitypass COMMAND ./test_reachabilitypass)

#
# test_combingpass
#

revng_add_test_executable(test_combingpass "${SRC}/CombingPass.cpp")
target_compile_definitions(test_combingpass PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_combingpass PRIVATE "${CMAKE_SOURCE_DIR}"
                                                    "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_combingpass
  revngcRestructureCFGPass
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_combingpass COMMAND ./test_combingpass --
                                       "${SRC}/TestGraphs/")

#
# test_vma
#

revng_add_test_executable(test_vma "${SRC}/ValueManipulationAnalysis.cpp")
target_compile_definitions(test_vma PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_vma PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_vma revngcValueManipulationAnalysis revng::revngModel
  revng::revngSupport Boost::unit_test_framework ${LLVM_LIBRARIES})
add_test(NAME test_vma COMMAND ./test_vma)
set_tests_properties(test_vma PROPERTIES LABELS "unit")

#
# dla_step_manager
#

revng_add_test_executable(dla_step_manager "${SRC}/DLAStepManager.cpp")
target_compile_definitions(dla_step_manager PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(dla_step_manager PRIVATE "${CMAKE_SOURCE_DIR}"
                                                    "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  dla_step_manager
  revngcDecompiler
  clangSerialization
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME dla_step_manager COMMAND ./dla_step_manager)

#
# MarkForSerializationTest
#

revng_add_test_executable(MarkForSerializationTest
                          "${SRC}/MarkForSerializationTest.cpp")
target_compile_definitions(MarkForSerializationTest
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(
  MarkForSerializationTest PRIVATE "${CMAKE_SOURCE_DIR}"
                                   "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  MarkForSerializationTest
  revngcDecompiler
  clangSerialization
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME MarkForSerializationTest COMMAND ./MarkForSerializationTest)

#
# DLASteps
#

revng_add_test_executable(dla_steps "${SRC}/DLASteps.cpp")
target_compile_definitions(dla_steps PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(dla_steps PRIVATE "${CMAKE_SOURCE_DIR}"
                                             "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  dla_steps
  revngcDecompiler
  revngcDataLayoutAnalysis
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME dla_steps COMMAND ./dla_steps)
