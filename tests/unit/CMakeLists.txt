#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_policy(SET CMP0060 NEW)

set(SRC "${CMAKE_SOURCE_DIR}/tests/unit")

add_subdirectory(llvm_lit_tests)
add_subdirectory(mlir_lit_tests)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

#
# test_combingpass
#

revng_add_test_executable(test_combingpass "${SRC}/CombingPass.cpp")
target_compile_definitions(test_combingpass PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_combingpass PRIVATE "${CMAKE_SOURCE_DIR}"
                                                    "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_combingpass
  revngcRestructureCFG
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_combingpass COMMAND test_combingpass -- "${SRC}/TestGraphs/")

#
# test_dla_step_manager
#

revng_add_test_executable(test_dla_step_manager "${SRC}/DLAStepManager.cpp")
target_compile_definitions(test_dla_step_manager
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(
  test_dla_step_manager PRIVATE "${CMAKE_SOURCE_DIR}" "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_dla_step_manager
  revngcDataLayoutAnalysis
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_dla_step_manager COMMAND test_dla_step_manager)

#
# test_mark_assignments_test
#

revng_add_test_executable(test_mark_assignments_test
                          "${SRC}/MarkAssignmentsTest.cpp")
target_compile_definitions(test_mark_assignments_test
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(
  test_mark_assignments_test PRIVATE "${CMAKE_SOURCE_DIR}"
                                     "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_mark_assignments_test
  revngcMarkAssignments
  revngcSupport
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_mark_assignments_test COMMAND test_mark_assignments_test)

#
# test_dla_steps
#

revng_add_test_executable(test_dla_steps "${SRC}/DLASteps.cpp")
target_compile_definitions(test_dla_steps PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_dla_steps PRIVATE "${CMAKE_SOURCE_DIR}"
                                                  "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_dla_steps
  revngcDataLayoutAnalysis
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_dla_steps COMMAND test_dla_steps)

#
# test_clift
#

revng_add_test_executable(test_clift "${SRC}/Clift.cpp")
target_compile_definitions(test_clift PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_clift PRIVATE "${CMAKE_SOURCE_DIR}"
                                              "${Boost_INCLUDE_DIRS}")
target_link_libraries(test_clift MLIRCliftDialect Boost::unit_test_framework
                      revng::revngUnitTestHelpers ${LLVM_LIBRARIES})
add_test(NAME test_clift COMMAND test_clift)
