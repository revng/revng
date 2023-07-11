#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
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
# test_vma
#

revng_add_test_executable(test_vma "${SRC}/ValueManipulationAnalysis.cpp")
target_compile_definitions(test_vma PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_vma PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_vma revngcValueManipulationAnalysis revng::revngModel
  revng::revngSupport Boost::unit_test_framework ${LLVM_LIBRARIES})
add_test(NAME test_vma COMMAND test_vma)
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
  revngcDataLayoutAnalysis
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME dla_step_manager COMMAND dla_step_manager)

#
# MarkAssignmentsTest
#

revng_add_test_executable(MarkAssignmentsTest "${SRC}/MarkAssignmentsTest.cpp")
target_compile_definitions(MarkAssignmentsTest PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(MarkAssignmentsTest PRIVATE "${CMAKE_SOURCE_DIR}"
                                                       "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  MarkAssignmentsTest
  revngcMarkAssignments
  revngcSupport
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME MarkAssignmentsTest COMMAND MarkAssignmentsTest)

#
# DLASteps
#

revng_add_test_executable(dla_steps "${SRC}/DLASteps.cpp")
target_compile_definitions(dla_steps PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(dla_steps PRIVATE "${CMAKE_SOURCE_DIR}"
                                             "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  dla_steps
  revngcDataLayoutAnalysis
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME dla_steps COMMAND dla_steps)

#
# Clift
#

revng_add_test_executable(clitTest "${SRC}/Clift.cpp")
target_compile_definitions(clitTest PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(clitTest PRIVATE "${CMAKE_SOURCE_DIR}"
                                            "${Boost_INCLUDE_DIRS}")
target_link_libraries(clitTest MLIRCliftDialect Boost::unit_test_framework
                      revng::revngUnitTestHelpers ${LLVM_LIBRARIES})
add_test(NAME clitTest COMMAND clitTest)
