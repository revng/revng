add_library(fakePipeboxImpl SHARED fakePipeboxImpl.cpp)
add_library(fakePipebox SHARED fakePipebox.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support Core)
target_link_libraries(fakePipeboxImpl ${LLVM_LIBRARIES} revngSupport revngModel)
target_link_libraries(fakePipebox fakePipeboxImpl ${Python_LIBRARIES}
                      ${LLVM_LIBRARIES} nanobind revngSupport)

target_include_directories(fakePipebox PUBLIC ${Python_INCLUDE_DIRS})

revng_add_test(
  NAME pypeline-test COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/python_cpp_test.py"
  "${CMAKE_CURRENT_BINARY_DIR}/libfakePipebox.so")
set_tests_properties(
  pypeline-test
  PROPERTIES LABELS "pypeline;test" ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}")
revng_add_test(
  NAME
  pypeline-annotations-test
  COMMAND
  "${CMAKE_CURRENT_SOURCE_DIR}/nanobind_test_annotations.sh"
  "${CMAKE_SOURCE_DIR}"
  -l
  "${CMAKE_CURRENT_BINARY_DIR}/libfakePipebox.so")
set_tests_properties(
  pypeline-annotations-test
  PROPERTIES LABELS "pypeline;test" ENVIRONMENT
             "PYTHONPATH=${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}")
