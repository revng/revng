#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

#
# Generate libmockPipeboxImpl and libmockPipebox
#
add_library(mockPipeboxImpl SHARED MockPipeboxImpl.cpp)
add_library(mockPipebox SHARED MockPipebox.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(mockPipeboxImpl ${LLVM_LIBRARIES} revngSupport revngModel)
target_link_libraries(
  mockPipebox
  mockPipeboxImpl
  ${Python3_LIBRARIES}
  ${LLVM_LIBRARIES}
  nanobind
  revngSupport
  revngClift)

target_include_directories(mockPipebox PUBLIC ${Python3_INCLUDE_DIRS})

configure_file(pipebox.py "${CMAKE_CURRENT_BINARY_DIR}" COPYONLY)

#
# Add tests that use libmockPipebox
#
revng_add_test(
  NAME pypeline-cpp-test COMMAND
  "${CMAKE_CURRENT_SOURCE_DIR}/python_cpp_test.py"
  "${CMAKE_CURRENT_BINARY_DIR}/libmockPipebox.so")
set_tests_properties(
  pypeline-cpp-test
  PROPERTIES
    LABELS "pypeline;test" ENVIRONMENT
    "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}:$ENV{PYTHONPATH}")

revng_add_test(
  NAME
  pypeline-annotations-test
  COMMAND
  "${CMAKE_CURRENT_SOURCE_DIR}/nanobind_test_annotations.sh"
  "${CMAKE_SOURCE_DIR}"
  -l
  "${CMAKE_CURRENT_BINARY_DIR}/libmockPipebox.so")
set_tests_properties(
  pypeline-annotations-test
  PROPERTIES
    LABELS "pypeline;test" ENVIRONMENT
    "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}:$ENV{PYTHONPATH}")

revng_add_test(
  NAME pypeline-native-test COMMAND
  "${CMAKE_CURRENT_SOURCE_DIR}/test_native_runners.sh" "${CMAKE_BINARY_DIR}")
set_tests_properties(
  pypeline-native-test
  PROPERTIES LABELS
             "pypeline;test"
             ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH};\
PYPELINE_PIPEBOX=${CMAKE_CURRENT_BINARY_DIR}/pipebox.py;\
PYPELINE_PIPELINE=${CMAKE_CURRENT_SOURCE_DIR}/pipeline.yml")
