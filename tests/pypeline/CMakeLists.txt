#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_test(
  NAME
  test_pypeline
  COMMAND
  "${Python_EXECUTABLE}"
  -m
  pytest
  -p
  no:cacheprovider
  -v
  --rootdir
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_all.py")
set_tests_properties(
  test_pypeline
  PROPERTIES LABELS "pypeline" ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}")

#
# Generate libmockPipeboxImpl and libmockPipebox
#
add_library(mockPipeboxImpl SHARED MockPipeboxImpl.cpp)
add_library(mockPipebox SHARED MockPipebox.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(mockPipeboxImpl ${LLVM_LIBRARIES} revngSupport revngModel)
target_link_libraries(mockPipebox mockPipeboxImpl ${Python_LIBRARIES}
                      ${LLVM_LIBRARIES} nanobind revngSupport)

target_include_directories(mockPipebox PUBLIC ${Python_INCLUDE_DIRS})

#
# Add tests that use libmockPipebox
#
revng_add_test(
  NAME pypeline-cpp-test COMMAND
  "${CMAKE_CURRENT_SOURCE_DIR}/python_cpp_test.py"
  "${CMAKE_CURRENT_BINARY_DIR}/libmockPipebox.so")
set_tests_properties(
  pypeline-cpp-test
  PROPERTIES LABELS "pypeline;test" ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}")

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
  PROPERTIES LABELS "pypeline;test" ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}")
